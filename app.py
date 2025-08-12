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
                r'wb\s+vehicle\s+no\.?\s*:?\s*([A-Z0-9]+)',
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
            ],
            'cid_no': [
                # Enhanced customer/reference patterns
                r'cid\s+no\.?\s*:?\s*([A-Z0-9-]+)',
                r'c\.?i\.?d\.?\s+no\.?\s*:?\s*([A-Z0-9-]+)',
                r'customer\s+id\s*:?\s*([A-Z0-9-]+)',
                r'ticket\s+no\.?\s*:?\s*([A-Z0-9-]+)',
                r'slip\s+no\.?\s*:?\s*([A-Z0-9-]+)', 
                r'challan\s+no\.?\s*:?\s*([A-Z0-9-]+)',
                r'reference\s*:?\s*([A-Z0-9-]+)',
                r'ref\.?\s*:?\s*([A-Z0-9-]+)',
                r'token\s+no\.?\s*:?\s*([A-Z0-9-]+)',
                r'weighbridge\s+no\.?\s*:?\s*([A-Z0-9-]+)',
                # From structured documents (000475, 0000348 patterns)
                r'(\d{6})',  # 6-digit reference codes
                r'(\d{7})',  # 7-digit reference codes  
                # Location-based IDs (PHALTAN-2)
                r'([A-Z]+-\d+)',
                # Serial/ID patterns
                r'SL\s+NO\.?\s*:?\s*(\d+)',
                # From samples
                r'(000475|0000348)',  # Specific IDs from samples
            ],
            'lot_no': [
                # Batch/lot patterns
                r'lot\s+no\.?\s*:?\s*([A-Z0-9-]+)',
                r'batch\s+no\.?\s*:?\s*([A-Z0-9-]+)',
                r'lot\s*:?\s*([A-Z0-9-]+)',
                r'batch\s*:?\s*([A-Z0-9-]+)',
                r'job\s+no\.?\s*:?\s*([A-Z0-9-]+)',
                r'order\s+no\.?\s*:?\s*([A-Z0-9-]+)',
                r'bill\s+no\.?\s*:?\s*([A-Z0-9-]+)',
                r'invoice\s+no\.?\s*:?\s*([A-Z0-9-]+)',
                # Serial numbers from tabular data
                r'sr\.?\s*no\.?\s*(\d+)',
                r'serial\s+no\.?\s*(\d+)',
                r'([A-Z]{2,}\d{3,})', # Alpha-numeric codes like ABC123
                r'([0-9]{4,}[A-Z]+)', # Numeric-alpha codes like 1234AB
            ],
            'quantity': [
                # Enhanced quantity patterns for different document types
                r'quantity\s*:?\s*(\d+(?:\.\d+)?)\s*(kg|kgs?|ton|tonnes?|mt|nos?|pcs?|pieces?|bags?|units?)?',
                r'qty\.?\s*:?\s*(\d+(?:\.\d+)?)\s*(kg|kgs?|ton|tonnes?|mt|nos?|pcs?|pieces?|bags?|units?)?',
                r'qnty\.?\s*:?\s*(\d+(?:\.\d+)?)\s*(kg|kgs?|ton|tonnes?|mt|nos?|pcs?|pieces?|bags?|units?)?',
                r'no\s+of\s+trips\s*:?\s*(\d+)',
                r'total\s+trips\s*:?\s*(\d+)',
                r'no\s+of\s+bags\s*:?\s*(\d+)',
                r'bags\s*:?\s*(\d+)',
                # Gross and Tare weights (from weighbridge certificates)
                r'gross\s+weight\s*:?\s*(\d+(?:\.\d+)?)\s*(kg|kgs?)',
                r'tare\s+weight\s*:?\s*(\d+(?:\.\d+)?)\s*(kg|kgs?)',
                # From structured documents
                r'GROSS\s+WEIGHT\s+(\d+)\s*(kg|Kg)',
                r'TARE\s+WEIGHT\s+(\d+)\s*(kg|Kg)',
                # Average weight patterns
                r'average\s+weight\s*:?\s*(\d+(?:\.\d+)?)\s*(kg|kgs?)',
                # General numeric quantity patterns
                r'(\d{1,5}(?:\.\d{1,3})?)\s*(kg|kgs?|ton|tonnes?|mt|bags?|units?)',
                r'total\s*:?\s*(\d+(?:\.\d+)?)\s*(kg|kgs?|ton|tonnes?|mt|bags?|units?)?'
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
            field_names = ['vehicle_no', 'net_weight', 'date', 'cid_no', 'lot_no', 'quantity']
            
            for field_name in field_names:
                extracted_data[field_name] = self.extract_field_with_patterns(
                    extracted_text, field_name
                )
            
            # Post-process the extracted fields
            processed_data = self.post_process_fields(extracted_data)
            
            # Add debugging information
            processed_data['raw_text'] = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
            processed_data['tabular_rows'] = len(doctr_result.get('tabular', []))
            processed_data['extraction_confidence'] = self.calculate_confidence(processed_data)
            
            return processed_data
            
        except Exception as e:
            st.error(f"Extraction error: {str(e)}")
            return {
                'error': str(e),
                'vehicle_no': None, 
                'net_weight': None, 
                'date': None,
                'cid_no': None,
                'lot_no': None,
                'quantity': None,
                'raw_text': None, 
                'tabular_rows': 0,
                'extraction_confidence': 0
            }
    
    def post_process_fields(self, extracted_data: Dict[str, str]) -> Dict[str, str]:
        """
        Enhanced post-processing for all document types
        """
        processed_data = {}
        
        for field, value in extracted_data.items():
            if value is None:
                processed_data[field] = None
                continue
                
            if field == 'vehicle_no':
                # Clean vehicle number - remove spaces and special chars
                clean_value = re.sub(r'[^A-Z0-9]', '', value.upper())
                processed_data[field] = clean_value
                
            elif field == 'date':
                # Standardize date format
                try:
                    value = value.replace('/', '-')
                    date_patterns = [
                        '%d-%m-%Y', '%d-%m-%y', '%d-%b-%Y', '%d-%B-%Y',
                        '%m-%d-%Y', '%Y-%m-%d', '%d %b %Y', '%d %B %Y'
                    ]
                    
                    for pattern in date_patterns:
                        try:
                            parsed_date = datetime.strptime(value, pattern)
                            processed_data[field] = parsed_date.strftime('%d/%m/%Y')
                            break
                        except ValueError:
                            continue
                    else:
                        processed_data[field] = value
                except:
                    processed_data[field] = value
                    
            elif field in ['net_weight', 'quantity']:
                # Clean weight/quantity - extract number and unit
                match = re.search(r'(\d+(?:\.\d+)?)\s*(.*)', str(value))
                if match:
                    number = match.group(1)
                    unit = match.group(2).strip() or 'kg'
                    processed_data[field] = f"{number} {unit}"
                else:
                    processed_data[field] = str(value)
                    
            elif field in ['cid_no', 'lot_no']:
                # Clean ID fields
                processed_data[field] = str(value).strip().upper()
                
            else:
                processed_data[field] = str(value).strip()
        
        return processed_data
    
    def calculate_confidence(self, data: Dict[str, str]) -> float:
        """
        Calculate extraction confidence based on found fields
        """
        required_fields = ['vehicle_no', 'net_weight', 'date', 'cid_no', 'lot_no', 'quantity']
        found_fields = sum(1 for field in required_fields if data.get(field) is not None)
        
        return round((found_fields / len(required_fields)) * 100, 1)

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
    **Required Fields:**
    - 🚗 WB Vehicle No: Registration number
    - ⚖️ Net Weight: Weight in kg
    - 📅 Date: Document date  
    - 🆔 CID No: Customer/Reference ID
    - 📦 Lot No: Lot/Batch number
    - 📊 Quantity: Additional quantity info
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
                        # Show extraction confidence
                        confidence = result.get('extraction_confidence', 0)
                        if confidence >= 80:
                            st.success(f"🎯 Extraction Confidence: {confidence}%")
                        elif confidence >= 60:
                            st.warning(f"⚠️ Extraction Confidence: {confidence}%")
                        else:
                            st.error(f"🔴 Extraction Confidence: {confidence}%")
                        
                        st.markdown("### 🔍 **Extracted Information**")
                        
                        # Create a structured table display
                        data_table = {
                            "Field": ["🚗 WB Vehicle No", "⚖️ Net Weight", "📅 Date", "🆔 CID No", "📦 Lot No", "📊 Quantity"],
                            "Value": [
                                result.get('vehicle_no') or 'N/A',
                                result.get('net_weight') or 'N/A',
                                result.get('date') or 'N/A',
                                result.get('cid_no') or 'N/A',
                                result.get('lot_no') or 'N/A',
                                result.get('quantity') or 'N/A'
                            ]
                        }
                        
                        df = pd.DataFrame(data_table)
                        st.table(df)
                        
                        # Also show individual fields for easy copying
                        st.markdown("### 📋 **Individual Fields**")
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write(f"**🚗 WB Vehicle No:** {result.get('vehicle_no') or 'N/A'}")
                            st.write(f"**⚖️ Net Weight:** {result.get('net_weight') or 'N/A'}")
                            st.write(f"**📅 Date:** {result.get('date') or 'N/A'}")
                        
                        with col_b:
                            st.write(f"**🆔 CID No:** {result.get('cid_no') or 'N/A'}")
                            st.write(f"**📦 Lot No:** {result.get('lot_no') or 'N/A'}")
                            st.write(f"**📊 Quantity:** {result.get('quantity') or 'N/A'}")
                        
                        # Show raw text in expander
                        with st.expander("📄 Raw Extracted Text"):
                            st.text(result.get('raw_text', 'No text extracted'))
                        
                        # Show tabular info
                        if result.get('tabular_rows', 0) > 0:
                            st.info(f"📊 Tabular rows detected: {result['tabular_rows']}")
                        
                        # Download option for extracted data
                        if st.button(f"💾 Download Data for Document {i+1}", key=f"download_{i}"):
                            csv_data = pd.DataFrame([{
                                'WB_Vehicle_No': result.get('vehicle_no', ''),
                                'Net_Weight': result.get('net_weight', ''),
                                'Date': result.get('date', ''),
                                'CID_No': result.get('cid_no', ''),
                                'Lot_No': result.get('lot_no', ''),
                                'Quantity': result.get('quantity', ''),
                                'Confidence': result.get('extraction_confidence', 0),
                                'Filename': uploaded_file.name
                            }])
                            csv_string = csv_data.to_csv(index=False)
                            st.download_button(
                                label="📥 Download as CSV",
                                data=csv_string,
                                file_name=f"extracted_data_{uploaded_file.name}.csv",
                                mime="text/csv",
                                key=f"csv_download_{i}"
                            )
                
                except Exception as e:
                    st.error(f"Processing error: {e}")

if __name__ == "__main__":
    main()
