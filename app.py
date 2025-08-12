import streamlit as st
import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import tempfile
import os
from PIL import Image
import pandas as pd

class VehicleDocumentExtractor:
    def __init__(self, model_name: str = 'crnn_vgg16_bn'):
        """
        Initialize the OCR extractor with docTR model - customized for vehicle weighbridge documents
        """
        if 'ocr_model' not in st.session_state:
            with st.spinner('Loading OCR model... This may take a moment.'):
                st.session_state.ocr_model = ocr_predictor(pretrained=True)
        self.model = st.session_state.ocr_model
        
        # Customized regex patterns based on the sample images
        self.patterns = {
            'vehicle_no': [
                # Indian vehicle number formats (from samples: IN2SC8478, MH31CB394)
                r'[A-Z]{2}\d{2}[A-Z]{1,2}\d{3,4}',  # Standard format
                r'IN\d{1,2}[A-Z]{1,2}\d{3,4}',      # IN prefix format
                r'MH\d{2}[A-Z]{1,2}\d{3,4}',        # MH prefix format
                r'vehicle\s+no\.?\s*:?\s*([A-Z0-9]+)',
                r'registration\s+no\.?\s*:?\s*([A-Z0-9]+)',
                r'reg\.?\s+no\.?\s*:?\s*([A-Z0-9]+)',
                r'veh\.?\s+no\.?\s*:?\s*([A-Z0-9]+)',
                # Direct extraction without labels
                r'([A-Z]{2}\d{2}[A-Z]{1,2}\d{3,4})'
            ],
            'net_weight': [
                # Weight patterns (from samples: 26605 kg, 20360 kg)
                r'net\s+weight\s*:?\s*(\d+)\s*(kg|kgs?)',
                r'(\d{4,6})\s*(kg|kgs?)',  # 4-6 digits followed by kg
                r'weight\s*:?\s*(\d+)\s*(kg|kgs?)',
                r'wt\.?\s*:?\s*(\d+)\s*(kg|kgs?)',
                r'nwt\.?\s*:?\s*(\d+)\s*(kg|kgs?)',
                # Weighbridge specific patterns
                r'(\d+)\s*kg',
                r'net\s*(\d+)\s*kg'
            ],
            'date': [
                # Date patterns (from samples: 17-Jan-2020, 13-11-2009)
                r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',           # DD-MM-YYYY or DD/MM/YYYY
                r'\d{1,2}[-/][A-Za-z]{3}[-/]\d{4}',       # DD-MMM-YYYY
                r'\d{1,2}\s+[A-Za-z]{3}\s+\d{4}',         # DD MMM YYYY
                r'date\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
                r'dated?\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
                r'dt\.?\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
                # Time patterns combined with date
                r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})\s*/?\s*\d{1,2}:\d{2}',
                # Specific formats from samples
                r'(\d{1,2}-[A-Za-z]{3}-\d{4})',
                r'(\d{1,2}-\d{1,2}-\d{4})'
            ],
            'cid_no': [
                # Customer/Challan ID patterns
                r'cid\s+no\.?\s*:?\s*([A-Z0-9]+)',
                r'c\.?i\.?d\.?\s+no\.?\s*:?\s*([A-Z0-9]+)',
                r'customer\s+id\s*:?\s*([A-Z0-9]+)',
                r'challan\s+id\s*:?\s*([A-Z0-9]+)',
                r'cid\s*:?\s*([A-Z0-9]+)',
                r'reference\s*:?\s*([A-Z0-9]+)',
                r'ref\.?\s*:?\s*([A-Z0-9]+)',
                # From sample: Reference 000475
                r'reference\s*:?\s*(\d{6})',
                r'(\d{6})',  # 6-digit reference numbers
            ],
            'lot_no': [
                # Lot/Batch number patterns
                r'lot\s+no\.?\s*:?\s*([A-Z0-9]+)',
                r'batch\s+no\.?\s*:?\s*([A-Z0-9]+)',
                r'lot\s*:?\s*([A-Z0-9]+)',
                r'batch\s*:?\s*([A-Z0-9]+)',
                r'lot\s+&\s+time\s+in\s*:?\s*([A-Z0-9\-\s:]+)',
                # Weighbridge specific
                r'slip\s+no\.?\s*:?\s*([A-Z0-9]+)',
                r'token\s+no\.?\s*:?\s*([A-Z0-9]+)',
                # From weighbridge sample
                r'([A-Z0-9]{6,})'  # Alphanumeric codes
            ],
            'quantity': [
                # Quantity patterns - may overlap with weight
                r'quantity\s*:?\s*(\d+(?:\.\d+)?)\s*(kg|kgs?|ton|tonnes?|mt|nos?|pcs?|pieces?)?',
                r'qty\.?\s*:?\s*(\d+(?:\.\d+)?)\s*(kg|kgs?|ton|tonnes?|mt|nos?|pcs?|pieces?)?',
                r'qnty\.?\s*:?\s*(\d+(?:\.\d+)?)\s*(kg|kgs?|ton|tonnes?|mt|nos?|pcs?|pieces?)?',
                r'quan\.?\s*:?\s*(\d+(?:\.\d+)?)\s*(kg|kgs?|ton|tonnes?|mt|nos?|pcs?|pieces?)?',
                # Weighbridge patterns (Tare weight, Gross weight)
                r'tare\s+weight\s*:?\s*(\d+)\s*(kg|kgs?)',
                r'gross\s+weight\s*:?\s*(\d+)\s*(kg|kgs?)',
                # From samples - specific weight readings
                r'(\d{3,5})\s*kg'
            ]
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing for weighbridge documents
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def extract_text_with_doctr(self, image_path: str) -> Tuple[str, Dict]:
        """
        Extract text from image using docTR with enhanced processing
        """
        # Load document
        doc = DocumentFile.from_images(image_path)
        
        # Perform OCR
        result = self.model(doc)
        
        # Extract text with positional information
        full_text = ""
        structured_text = []
        
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = ""
                    for word in line.words:
                        line_text += word.value + " "
                        structured_text.append({
                            'text': word.value,
                            'confidence': word.confidence,
                            'geometry': word.geometry
                        })
                    full_text += line_text.strip() + "\n"
        
        return full_text.strip(), {'result': result, 'structured': structured_text}
    
    def extract_field_with_patterns(self, text: str, field_name: str) -> Optional[str]:
        """
        Enhanced field extraction with confidence scoring
        """
        if field_name not in self.patterns:
            return None
        
        text_clean = re.sub(r'[^\w\s\-:/.]', ' ', text)  # Clean special chars
        text_lines = text_clean.split('\n')
        
        candidates = []
        
        for pattern in self.patterns[field_name]:
            # Search in full text
            matches = re.findall(pattern, text_clean, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1] if len(match) > 1 else ''
                if match and len(match.strip()) > 0:
                    candidates.append(match.strip())
            
            # Search line by line for better context
            for line in text_lines:
                line_matches = re.findall(pattern, line, re.IGNORECASE)
                for match in line_matches:
                    if isinstance(match, tuple):
                        match = match[0] if match[0] else match[1] if len(match) > 1 else ''
                    if match and len(match.strip()) > 0:
                        candidates.append(match.strip())
        
        if not candidates:
            return None
        
        # Post-process candidates based on field type
        if field_name == 'vehicle_no':
            # Prefer alphanumeric with correct format
            for candidate in candidates:
                clean_candidate = re.sub(r'\s+', '', candidate.upper())
                if re.match(r'[A-Z]{2}\d{2}[A-Z]{1,2}\d{3,4}', clean_candidate):
                    return clean_candidate
            return candidates[0].upper().replace(' ', '')
        
        elif field_name == 'net_weight':
            # Prefer larger weights (net weight typically higher than tare)
            weight_candidates = []
            for candidate in candidates:
                weight_match = re.search(r'(\d+)', candidate)
                if weight_match:
                    weight_val = int(weight_match.group(1))
                    if weight_val > 1000:  # Reasonable net weight threshold
                        weight_candidates.append((weight_val, candidate))
            
            if weight_candidates:
                weight_candidates.sort(key=lambda x: x[0], reverse=True)
                return weight_candidates[0][1]
            return candidates[0] if candidates else None
        
        elif field_name == 'date':
            # Prefer complete date formats
            for candidate in candidates:
                if re.search(r'\d{4}', candidate):  # Has year
                    return candidate
            return candidates[0] if candidates else None
        
        elif field_name == 'cid_no':
            # Prefer numeric IDs
            for candidate in candidates:
                if re.match(r'\d+', candidate):
                    return candidate.upper()
            return candidates[0].upper() if candidates else None
        
        return candidates[0].upper() if field_name in ['vehicle_no', 'cid_no', 'lot_no'] else candidates[0]
    
    def post_process_fields(self, extracted_data: Dict[str, str]) -> Dict[str, str]:
        """
        Enhanced post-processing for weighbridge documents
        """
        processed_data = {}
        
        for field, value in extracted_data.items():
            if value is None:
                processed_data[field] = None
                continue
                
            if field == 'vehicle_no':
                # Clean and format vehicle number
                clean_value = re.sub(r'[^A-Z0-9]', '', value.upper())
                processed_data[field] = clean_value
                
            elif field == 'date':
                # Enhanced date processing
                try:
                    # Handle various formats from samples
                    value = value.replace('/', '-')
                    
                    date_patterns = [
                        '%d-%m-%Y', '%d-%m-%y',
                        '%d-%b-%Y', '%d-%B-%Y',
                        '%m-%d-%Y', '%Y-%m-%d',
                        '%d %b %Y', '%d %B %Y'
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
                # Extract weight/quantity with units
                match = re.search(r'(\d+(?:\.\d+)?)\s*(.*)', str(value))
                if match:
                    number = match.group(1)
                    unit = match.group(2).strip() or 'kg'
                    processed_data[field] = f"{number} {unit}"
                else:
                    processed_data[field] = str(value)
                    
            elif field == 'cid_no':
                # Clean CID/Reference numbers
                clean_value = re.sub(r'[^A-Z0-9]', '', str(value).upper())
                processed_data[field] = clean_value
                
            else:
                processed_data[field] = str(value).strip()
        
        return processed_data
    
    def extract_from_image(self, image_path: str, preprocess: bool = True) -> Dict[str, str]:
        """
        Main extraction method optimized for weighbridge documents
        """
        try:
            # Extract text using docTR
            extracted_text, doctr_result = self.extract_text_with_doctr(image_path)
            
            # Extract each field
            extracted_data = {}
            field_names = ['vehicle_no', 'net_weight', 'date', 'cid_no', 'lot_no', 'quantity']
            
            for field_name in field_names:
                extracted_data[field_name] = self.extract_field_with_patterns(extracted_text, field_name)
            
            # Post-process the extracted fields
            processed_data = self.post_process_fields(extracted_data)
            
            # Add debugging information
            processed_data['raw_text'] = extracted_text
            processed_data['extraction_confidence'] = self.calculate_confidence(processed_data)
            
            return processed_data
            
        except Exception as e:
            return {
                'error': str(e),
                'vehicle_no': None,
                'net_weight': None,
                'date': None,
                'cid_no': None,
                'lot_no': None,
                'quantity': None,
                'raw_text': None,
                'extraction_confidence': 0
            }
    
    def calculate_confidence(self, data: Dict[str, str]) -> float:
        """
        Calculate overall extraction confidence
        """
        total_fields = 6  # vehicle_no, net_weight, date, cid_no, lot_no, quantity
        extracted_fields = sum(1 for field in ['vehicle_no', 'net_weight', 'date', 'cid_no', 'lot_no', 'quantity'] 
                             if data.get(field) is not None)
        return round((extracted_fields / total_fields) * 100, 1)

def main():
    st.set_page_config(
        page_title="Vehicle Weighbridge OCR Extractor",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Vehicle Weighbridge Document OCR Extractor")
    st.markdown("Upload weighbridge/vehicle documents to extract key information using AI-powered OCR")
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    preprocess_image = st.sidebar.checkbox("Apply Image Preprocessing", value=True, 
                                         help="Enhances image quality for better OCR results")
    
    st.sidebar.markdown("### 📋 Extracted Fields:")
    st.sidebar.markdown("""
    - **Vehicle No**: Registration number
    - **Net Weight**: Weight in kg
    - **Date**: Document date  
    - **CID No**: Customer/Reference ID
    - **Lot No**: Lot/Batch number
    - **Quantity**: Additional weight info
    """)
    
    # Initialize extractor
    @st.cache_resource
    def load_extractor():
        return VehicleDocumentExtractor()
    
    extractor = load_extractor()
    
    # File uploader
    uploaded_files = st.file_uploader(
        "📁 Choose weighbridge document images",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        accept_multiple_files=True,
        help="Upload clear images of weighbridge slips or vehicle documents"
    )
    
    if uploaded_files:
        for i, uploaded_file in enumerate(uploaded_files):
            st.subheader(f"📄 Document {i+1}: {uploaded_file.name}")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🖼️ Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
                
                # Image info
                st.caption(f"Size: {image.size[0]}x{image.size[1]} pixels")
            
            with col2:
                st.subheader("🔍 Extracted Information")
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image.save(tmp_file.name)
                    
                    # Process with OCR
                    with st.spinner('🤖 Analyzing document with AI OCR...'):
                        result = extractor.extract_from_image(tmp_file.name, preprocess_image)
                    
                    # Clean up temp file
                    os.unlink(tmp_file.name)
                
                # Display results
                if result.get('error'):
                    st.error(f"❌ Error: {result['error']}")
                else:
                    # Confidence score
                    confidence = result.get('extraction_confidence', 0)
                    if confidence >= 80:
                        st.success(f"🎯 Extraction Confidence: {confidence}%")
                    elif confidence >= 60:
                        st.warning(f"⚠️ Extraction Confidence: {confidence}%")
                    else:
                        st.error(f"🔴 Extraction Confidence: {confidence}%")
                    
                    # Create a clean display of extracted data
                    extracted_info = {
                        "🚗 Vehicle No": result['vehicle_no'] or "Not found",
                        "⚖️ Net Weight": result['net_weight'] or "Not found", 
                        "📅 Date": result['date'] or "Not found",
                        "🆔 CID No": result['cid_no'] or "Not found",
                        "📦 Lot No": result['lot_no'] or "Not found",
                        "📊 Quantity": result['quantity'] or "Not found"
                    }
                    
                    # Display in a structured format
                    for field, value in extracted_info.items():
                        if value != "Not found":
                            st.success(f"**{field}:** `{value}`")
                        else:
                            st.warning(f"**{field}:** {value}")
                    
                    # Results summary table
                    st.subheader("📋 Summary Table")
                    df_data = []
                    for field, value in extracted_info.items():
                        clean_field = field.split(' ', 1)[1]  # Remove emoji
                        df_data.append({"Field": clean_field, "Value": value, "Status": "✅" if value != "Not found" else "❌"})
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Raw text expander
                    with st.expander("🔍 View Raw Extracted Text (Debug)"):
                        st.text_area("Raw OCR Output", result['raw_text'], height=200)
                    
                    # Download options
                    col_json, col_csv = st.columns(2)
                    
                    with col_json:
                        json_str = json.dumps(result, indent=2)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_str,
                            file_name=f"extracted_data_{uploaded_file.name}.json",
                            mime="application/json"
                        )
                    
                    with col_csv:
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV", 
                            data=csv_data,
                            file_name=f"extracted_data_{uploaded_file.name}.csv",
                            mime="text/csv"
                        )
            
            st.divider()
    
    else:
        st.info("Upload weighbridge document images to get started!")

if __name__ == "__main__":
    main()
