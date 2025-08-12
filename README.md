# **AI-Powered Document Data Extraction System for Weighbridge Slips**

This project is an **end-to-end OCR and reasoning pipeline** designed to extract structured data from weighbridge slip images.  
It uses **docTR** for high-accuracy OCR and **Qwen 3** for intelligent reasoning to produce clean JSON output.

---

## Features
- Extracts:
  - WB Vehicle No
  - Net Weight
  - Date
  - CID No
  - Lot No
  - Quantity
- Works with **JPG, PNG, PDF** files.
- Automatic **image preprocessing** (resize, deskew, denoise, convert to RGB).
- **REST API** built with FastAPI for easy integration.
- JSON output ready for database storage or further processing.
- Plug-and-play integration with enterprise platforms like **Nexial**.

---

## Workflow
1. **Upload**  
   User uploads a weighbridge slip image via API.

2. **Preprocessing**  
   OpenCV enhances image quality for better OCR results.

3. **OCR Extraction** *(docTR)*  
   - Detection model: `db_resnet50`  
   - Recognition model: `crnn_vgg16_bn`  
   - Produces layout-aware raw text.

4. **Reasoning & Structuring** *(Qwen 3)*  
   - Takes OCR text and extracts required fields using natural language instructions.  
   - Returns clean, structured JSON.

5. **Output**  
   API responds with extracted JSON and optional raw OCR text.

---

## 🛠 Tech Stack
- **Language:** Python 3.11+  
- **Framework:** FastAPI  
- **OCR Engine:** docTR  
- **Image Processing:** OpenCV  
- **LLM Reasoning:** Qwen 3 Instruct (via Hugging Face Transformers)  
- **Server:** Uvicorn  

---

## Project Structure
```
├── main.py              # FastAPI application
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
└── sample_images/       # Example weighbridge slips
```

---

## Future Improvements
- Support for handwriting recognition.
- Auto-detection of document type.
- Integration with cloud storage for automatic processing.
