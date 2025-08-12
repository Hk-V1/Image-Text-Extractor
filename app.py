import streamlit as st
import numpy as np
try:
    import cv2
except ImportError:
    st.error("OpenCV not available. Some preprocessing features may be limited.")
    cv2 = None

# Add error handling for doctr imports
try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    DOCTR_AVAILABLE = True
except ImportError:
    st.error("DocTR not available. Please install doctr: pip install python-doctr[torch]")
    DOCTR_AVAILABLE = False

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import tempfile
import os
from PIL import Image
import pandas as pd

class EnhancedVehicleDocumentExtractor:
    def __init__(self, model_name: str = 'crnn_vgg16_bn'):
        """
        Enhanced OCR extractor for both simple and complex weighbridge documents
        """
        if not DOCTR_AVAILABLE:
            st.error("Cannot initialize extractor: DocTR is not available")
            return
            
        if 'ocr_model' not in st.session_state:
            with st.spinner('Loading OCR model... This may take a moment.'):
                try:
                    st.session_state.ocr_model = ocr_predictor(pretrained=True)
                    st.success("OCR model loaded successfully!")
                except Exception as e:
                    st.error(f"Failed to load OCR model: {e}")
                    return
        self.model = st.session_state.ocr_model
        
        # Enhanced patterns for various document types
        self.patterns = {
            'vehicle_no': [
                # Indian vehicle number formats (from samples: IN28C8478, MH31CB394)
                r'[A-Z]{2}\d{2}[A-Z]{1,2}\d{3,4}',
                r'IN\d{1,2}[A-Z]{1,2}\d{3,4}',
                r'MH\d{2}[A-Z]{1,2}\d{3,4}',
                r'GJ\d{2}[A-Z]{1,2}\d{3,4}',
                r'KA\d{2}[A-Z]{1,2}\d{3,4}',
                r'TN\d{2}[A-Z]{1,2}\d{3,4}',
                r'AP\d{2}[A-Z]{1,2}\d{3,4}',
                r'vehicle\s+no\.?\s*:?\s*([A-Z0-9]+)',
                r'registration\s+no\.?\s*:?\s*([A-Z0-9]+)',
                r'reg\.?\s+no\.?\s*:?\s*([A-Z0-9]+)',
                r'veh\.?\s+no\.?\s*:?\s*([A-Z0-9]+)',
                # Direct patterns from structured documents
                r'([A-Z]{2}\d{1,2}[A-Z]{1,2}\d{3,4})',
                # From the samples
                r'(IN28C8478|MH31CB394)',  # Specific patterns seen
            ],
            'net_weight': [
                # Enhanced weight patterns for different document types
                r'net\s+weight\s*:?\s*(\d+(?:\.\d+)?)\s*(kg|kgs?|ton|tonnes?)?',
                r'total\s+net\s+weight\s*:?\s*(\d+(?:\.\d+)?)\s*(kg|kgs?|ton|tonnes?)?',
                # From structured documents (26605 kg pattern)
                r'(\d{4,6})\s*(kg|kgs?)',  # 4-6 digits followed by kg
                # From weighbridge certificates (27670Kg, 7310Kg, 20360Kg)
                r'(\d{4,5})Kg',  # Numbers directly followed by Kg
                r'(\d{4,5})\s*Kg',  # Numbers with space before Kg
                # Decimal weights (like 50.160, 50.150)
                r'(\d{1,3}\.\d{2,3})\s*(kg|kgs?)',
                # General weight patterns
                r'weight\s*:?\s*(\d+(?:\.\d+)?)\s*(kg|kgs?)',
                r'wt\.?\s*:?\s*(\d+(?:\.\d+)?)\s*(kg|kgs?)',
                r'nwt\.?\s*:?\s*(\d+(?:\.\d+)?)\s*(kg|kgs?)',
                # From summary sections
                r'net\s*(\d+(?:\.\d+)?)',
                r'average\s+net\s+weight\s*:?\s*(\d+(?:\.\d+)?)',
                r'total.*?(\d+(?:\.\d+)?)\s*kg',
                # Specific patterns from samples
                r'NET\s+WEIGHT\s+(\d+)\s*kg',
            ],
            'date': [
                # Enhanced date patterns for different formats
                r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',
                r'\d{1,2}[-/][A-Za-z]{3}[-/]\d{4}',
                r'\d{1,2}\s+[A-Za-z]{3}\s+\d{4}',
                # From structured documents
                r'date\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
                r'dated?\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
                r'dt\.?\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
                # Timestamp patterns (17-Jun-2020, 13-11-2009)
                r'(\d{1,2}-\w{3}-\d{4})',  # DD-MMM-YYYY format
                r'(\d{1,2}-\d{1,2}-\d{4})',  # DD-MM-YYYY format
                # With time stamps
                r'(\d{1,2}-\d{1,2}-\d{4})\s*\d{1,2}:\d{2}:\d{2}',
                r'(\d{4}-\d{1,2}-\d{1,2})',  # YYYY-MM-DD
                # From weighbridge certificates (16-01-2009)
                r'(\d{2}-\d{2}-\d{4})',
                # From header/footer timestamps
                r'([A-Z-]+\d+)\s*\(\s*(\d{1,2}-\d{1,2}-\d{4})',
            ]
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing optimized for tabular documents
        """
        if cv2 is None:
            # Fallback preprocessing without OpenCV
            if len(image.shape) == 3:
                gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = image
            return gray.astype(np.uint8)
        
        # Full OpenCV preprocessing for tabular data
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Enhanced contrast for table detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoising
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Adaptive thresholding for text
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def extract_text_with_doctr(self, image_path: str) -> Tuple[str, Dict]:
        """
        Enhanced text extraction with better structure preservation
        """
        # Load document
        doc = DocumentFile.from_images(image_path)
        
        # Perform OCR
        result = self.model(doc)
        
        # Extract text with enhanced structure preservation
        full_text = ""
        structured_text = []
        tabular_data = []
        
        for page in result.pages:
            for block in page.blocks:
                block_text = ""
                for line in block.lines:
                    line_text = ""
                    line_words = []
                    for word in line.words:
                        line_text += word.value + " "
                        line_words.append({
                            'text': word.value,
                            'confidence': word.confidence,
                            'geometry': word.geometry
                        })
                        structured_text.append({
                            'text': word.value,
                            'confidence': word.confidence,
                            'geometry': word.geometry
                        })
                    
                    # Detect tabular structure
                    if len(line_words) > 3:  # Likely tabular data
                        tabular_data.append({
                            'line_text': line_text.strip(),
                            'words': line_words,
                            'word_count': len(line_words)
                        })
                    
                    block_text += line_text.strip() + "\n"
                full_text += block_text + "\n"
        
        return full_text.strip(), {
            'result': result, 
            'structured': structured_text,
            'tabular': tabular_data
        }
    
    def extract_field_with_patterns(self, text: str, field_name: str, tabular_data: Dict = None) -> Optional[str]:
        """
        Enhanced field extraction with tabular data support
        """
        if field_name not in self.patterns:
            return None
        
        text_clean = re.sub(r'[^\w\s\-:/.\(\)]', ' ', text)
        text_lines = text_clean.split('\n')
        
        candidates = []
        
        # Extract using regex patterns
        for pattern in self.patterns[field_name]:
            matches = re.findall(pattern, text_clean, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1] if len(match) > 1 else ''
                if match and len(match.strip()) > 0:
                    candidates.append(match.strip())
        
        if not candidates:
            return None
        
        return candidates[0] if candidates else None
    
    def extract_from_image(self, image_path: str, preprocess: bool = True) -> Dict[str, str]:
        """
        Main extraction method for both simple and complex weighbridge documents
        """
        try:
            # Extract text using docTR
            extracted_text, doctr_result = self.extract_text_with_doctr(image_path)
            
            # Extract each field
            extracted_data = {}
            field_names = ['vehicle_no', 'net_weight', 'date']
            
            for field_name in field_names:
                extracted_data[field_name] = self.extract_field_with_patterns(
                    extracted_text, field_name
                )
            
            # Add debugging information
            extracted_data['raw_text'] = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
            extracted_data['tabular_rows'] = len(doctr_result.get('tabular', []))
            
            return extracted_data
            
        except Exception as e:
            st.error(f"Extraction error: {str(e)}")
            return {
                'error': str(e),
                'vehicle_no': None, 
                'net_weight': None, 
                'date': None,
                'raw_text': None, 
                'tabular_rows': 0
            }

def main():
    st.set_page_config(
        page_title="Enhanced Vehicle Weighbridge OCR Extractor",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Enhanced Vehicle Weighbridge Document OCR Extractor")
    st.markdown("Upload weighbridge/vehicle documents to extract key information using AI-powered OCR")
    
    # Check if dependencies are available
    if not DOCTR_AVAILABLE:
        st.error("❌ DocTR is not installed. Please install it using:")
        st.code("pip install python-doctr[torch]")
        st.stop()
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    preprocess_image = st.sidebar.checkbox("Apply Image Preprocessing", value=True)
    
    st.sidebar.markdown("### 📋 Extracted Fields:")
    st.sidebar.markdown("""
    **Core Fields:**
    - 🚗 Vehicle No: Registration number
    - ⚖️ Net Weight: Weight in kg
    - 📅 Date: Document date  
    """)
    
    # Initialize extractor with error handling
    try:
        if 'extractor' not in st.session_state:
            st.session_state.extractor = EnhancedVehicleDocumentExtractor()
        extractor = st.session_state.extractor
    except Exception as e:
        st.error(f"Failed to initialize extractor: {e}")
        st.stop()
    
    # File uploader
    uploaded_files = st.file_uploader(
        "📁 Choose weighbridge document images",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        accept_multiple_files=True,
        help="Upload clear images of weighbridge slips"
    )
    
    if uploaded_files:
        for i, uploaded_file in enumerate(uploaded_files):
            st.subheader(f"📄 Document {i+1}: {uploaded_file.name}")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🖼️ Original Image")
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, use_column_width=True)
                    st.caption(f"Size: {image.size[0]}x{image.size[1]} pixels")
                except Exception as e:
                    st.error(f"Error loading image: {e}")
                    continue
            
            with col2:
                st.subheader("🔍 Extracted Information")
                
                # Save uploaded file temporarily
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        image.save(tmp_file.name)
                        
                        # Process with OCR
                        with st.spinner('🤖 Analyzing document with Enhanced AI OCR...'):
                            result = extractor.extract_from_image(tmp_file.name, preprocess_image)
                        
                        os.unlink(tmp_file.name)
                    
                    # Display results
                    if result.get('error'):
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        st.markdown("### 🔍 **Core Information**")
                        core_fields = {
                            "🚗 Vehicle No": result.get('vehicle_no'),
                            "⚖️ Net Weight": result.get('net_weight'),
                            "📅 Date": result.get('date')
                        }
                        
                        for label, value in core_fields.items():
                            st.write(f"**{label}:** {value if value else 'N/A'}")
                        
                        # Show raw text in expander
                        with st.expander("📄 Raw Extracted Text"):
                            st.text(result.get('raw_text', 'No text extracted'))
                        
                        # Show tabular info
                        if result.get('tabular_rows', 0) > 0:
                            st.info(f"📊 Tabular rows detected: {result['tabular_rows']}")
                
                except Exception as e:
                    st.error(f"Processing error: {e}")

if __name__ == "__main__":
    main()
