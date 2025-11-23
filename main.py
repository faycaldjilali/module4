from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
import pandas as pd
from datetime import datetime, date
import json
import io
import tempfile
import os
import PyPDF2
import time
import re
import uvicorn
from typing import List, Optional
import asyncio

app = FastAPI(title="BOAMP Data Extractor")

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Store processing state (in production, use Redis or database)
processing_state = {}

def get_all_records_for_date(target_date: str, max_records: int = 10000):
    """Get all records for a specific date with all available fields"""
    url = "https://boamp-datadila.opendatasoft.com/api/explore/v2.1/catalog/datasets/boamp/records"
    all_records = []
    offset = 0
    limit = 100

    while len(all_records) < max_records:
        params = {
            'order_by': 'dateparution DESC',
            'limit': limit,
            'offset': offset
        }

        response = requests.get(url, params=params)

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Error {response.status_code}: {response.text}")

        data = response.json()
        records = data.get('results', [])

        if not records:
            break  # No more records

        # Filter records for our target date
        target_records = [record for record in records if record.get('dateparution') == target_date]

        # If we found target records, add them
        if target_records:
            all_records.extend(target_records)

        # Check if we've moved past our target date (since we're sorting DESC)
        if records and records[-1].get('dateparution', '') < target_date:
            break

        offset += limit

        if offset > 10000:
            break

    return all_records

def create_excel_simple(records: List[dict], target_date: str):
    """Simple and robust Excel creation"""
    cleaned_records = []
    for record in records:
        cleaned_record = {}
        for key, value in record.items():
            if isinstance(value, (list, dict)):
                cleaned_record[key] = json.dumps(value, ensure_ascii=False)
            elif value is None:
                cleaned_record[key] = ''
            else:
                cleaned_record[key] = value
        cleaned_records.append(cleaned_record)

    df = pd.DataFrame(cleaned_records)
    return df

def filter_by_keywords(df: pd.DataFrame, keywords: List[str]):
    """Filter DataFrame by keywords"""
    df_str = df.astype(str).apply(lambda x: x.str.lower())
    all_matches = pd.DataFrame()

    for keyword in keywords:
        mask = df_str.apply(lambda x: x.str.contains(keyword.lower(), na=False))
        filtered_df = df[mask.any(axis=1)]
        
        if not filtered_df.empty:
            filtered_df = filtered_df.copy()
            filtered_df["keyword"] = keyword
            all_matches = pd.concat([all_matches, filtered_df], ignore_index=True)

    return all_matches

def search_keywords_and_find_lot(text: str, keywords: List[str]):
    """
    Search for keywords in PDF text and find ALL lot numbers that appear before them
    """
    try:
        results = []
        
        # Search for each keyword
        for keyword in keywords:
            # Find all occurrences of the keyword
            keyword_matches = list(re.finditer(re.escape(keyword), text, re.IGNORECASE))
            
            for match in keyword_matches:
                keyword_position = match.start()
                
                # Extract more text before the keyword (look back up to 500 characters)
                text_before = text[max(0, keyword_position - 1000):keyword_position]
                
                # Improved lot pattern to catch more formats
                lot_patterns = [
                    r'(lot|LOT)\s*[:\-\s]*\s*(\d+[-\w]*)',  # lot: 123, LOT-456, lot 789
                    r'(Lot\s*\d+)',  # Lot 123
                    r'(lot\s*\d+)',  # lot 123
                    r'\b(\d+)\s*-\s*Lot',  # 123 - Lot
                    r'\b(LOT\s*[A-Z]*\d+)',  # LOT A123, LOT 456
                ]
                
                all_lot_matches = []
                
                for pattern in lot_patterns:
                    matches = re.findall(pattern, text_before, re.IGNORECASE)
                    for match_tuple in matches:
                        if isinstance(match_tuple, tuple):
                            # For patterns that capture groups
                            lot_number = match_tuple[1] if len(match_tuple) > 1 else match_tuple[0]
                        else:
                            # For patterns that capture directly
                            lot_number = match_tuple
                        
                        # Clean up the lot number
                        lot_number = re.sub(r'^(lot|LOT)\s*', '', lot_number, flags=re.IGNORECASE)
                        lot_number = lot_number.strip(' :-\t')
                        
                        if lot_number and lot_number not in [lm[0] for lm in all_lot_matches]:
                            all_lot_matches.append((lot_number, pattern))
                
                # Remove duplicates while preserving order
                unique_lots = []
                seen = set()
                for lot_num, pattern in all_lot_matches:
                    if lot_num not in seen:
                        seen.add(lot_num)
                        unique_lots.append(lot_num)
                
                if unique_lots:
                    for lot_number in unique_lots:
                        results.append({
                            'keyword': keyword,
                            'lot_number': lot_number
                        })
        
        return results
            
    except Exception as e:
        return []

def check_visite_obligatoire(text: str, keywords: List[str]):
    """
    Search for keywords in PDF text and check if 'visite' appears before them
    """
    try:
        # Search for each keyword
        for keyword in keywords:
            # Find all occurrences of the keyword
            keyword_matches = list(re.finditer(re.escape(keyword), text, re.IGNORECASE))
            
            for match in keyword_matches:
                keyword_position = match.start()
                
                # Extract text before the keyword (look back up to 500 characters)
                text_before = text[max(0, keyword_position - 500):keyword_position]
                
                # Check if "visite" appears before the keyword
                visite_patterns = [r"visites", r"visite"]
                
                for pattern in visite_patterns:
                    if re.search(pattern, text_before, re.IGNORECASE):
                        return "yes"
        
        return "no"
            
    except Exception as e:
        return "no"

def extract_pdf_content(df: pd.DataFrame, process_id: str):
    """Extract PDF content and analyze for lots and visite information"""
    df_with_pdf = df.copy()
    df_with_pdf['generated_link'] = ""
    df_with_pdf['pdf_content'] = ""
    df_with_pdf['pdf_status'] = ""
    df_with_pdf['pages_extracted'] = 0
    df_with_pdf['lot_numbers'] = ""
    df_with_pdf['visite_obligatoire'] = ""
    df_with_pdf['keywords_used'] = ""
    
    total_records = len(df_with_pdf)
    successful = 0
    errors = 0
    
    processing_state[process_id]['current_step'] = 'pdf_extraction'
    processing_state[process_id]['total_records'] = total_records
    processing_state[process_id]['processed_records'] = 0
    
    for index, row in df_with_pdf.iterrows():
        dateparution_str = row.get('dateparution')
        idweb = row.get('idweb', 'N/A')
        keywords_from_row = row.get('keyword', '')
        
        # Update progress
        processing_state[process_id]['processed_records'] = index + 1
        processing_state[process_id]['current_record'] = idweb
        
        if idweb == 'N/A':
            df_with_pdf.at[index, 'pdf_status'] = "Skipped - No ID"
            errors += 1
            continue
            
        try:
            # Parse date
            if isinstance(dateparution_str, str):
                date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']
                dateparution = None
                for fmt in date_formats:
                    try:
                        dateparution = datetime.strptime(dateparution_str, fmt)
                        break
                    except ValueError:
                        continue
                if dateparution is None:
                    df_with_pdf.at[index, 'pdf_status'] = "Error - Date parsing failed"
                    errors += 1
                    continue
            else:
                dateparution = dateparution_str
            
            # Generate link
            link = f"https://www.boamp.fr/telechargements/FILES/PDF/{dateparution.year}/{dateparution.month:02d}/{idweb}.pdf"
            
            # Add link to DataFrame
            df_with_pdf.at[index, 'generated_link'] = link
            
            # Store keywords used for this row
            df_with_pdf.at[index, 'keywords_used'] = str(keywords_from_row)
            
            # Download and extract PDF content
            try:
                # Download the PDF
                response = requests.get(link, timeout=30)
                response.raise_for_status()
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_file.write(response.content)
                    temp_path = temp_file.name
                
                # Extract text using PyPDF2
                with open(temp_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    
                    # Extract text from each page
                    full_text = ""
                    for page_num, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        full_text += f"Page {page_num + 1}:\n{page_text}\n\n"
                    
                    # Add PDF content to DataFrame
                    df_with_pdf.at[index, 'pdf_content'] = full_text
                    df_with_pdf.at[index, 'pages_extracted'] = len(pdf_reader.pages)
                    df_with_pdf.at[index, 'pdf_status'] = "Success"
                    
                    # Extract keywords from the row (could be string or list)
                    if isinstance(keywords_from_row, str):
                        # Split by semicolon if it's a combined string from deduplication
                        search_keywords = [k.strip() for k in keywords_from_row.split(';') if k.strip()]
                    else:
                        search_keywords = [str(keywords_from_row)]
                    
                    # Search for lot numbers
                    lot_results = search_keywords_and_find_lot(full_text, search_keywords)
                    if lot_results:
                        unique_lots = set()
                        for result in lot_results:
                            unique_lots.add(f"lot-{result['lot_number']}")
                        df_with_pdf.at[index, 'lot_numbers'] = ', '.join(sorted(unique_lots))
                    
                    # Check for visite obligatoire
                    visite_keywords = ["obligatoires", "obligatoire"]
                    visite_result = check_visite_obligatoire(full_text, visite_keywords)
                    df_with_pdf.at[index, 'visite_obligatoire'] = visite_result
                    
                    successful += 1
                
                # Clean up
                os.unlink(temp_path)
                
            except Exception as e:
                error_msg = f"Error processing PDF: {str(e)}"
                df_with_pdf.at[index, 'pdf_content'] = error_msg
                df_with_pdf.at[index, 'pdf_status'] = f"Error: {str(e)}"
                errors += 1
            
            # Add a small delay to be respectful to the server
            time.sleep(0.5)
            
        except Exception as e:
            error_msg = f"Error processing row: {str(e)}"
            df_with_pdf.at[index, 'pdf_content'] = error_msg
            df_with_pdf.at[index, 'pdf_status'] = f"Error: {str(e)}"
            errors += 1
            continue
    
    processing_state[process_id]['status'] = 'completed'
    processing_state[process_id]['result'] = df_with_pdf.to_dict('records')
    
    return df_with_pdf

def remove_duplicates(df: pd.DataFrame, id_column: str, keyword_column: str):
    """Remove duplicates from DataFrame by combining keywords"""
    # Group by ID and combine keywords
    def combine_keywords(group):
        if len(group) > 1:
            # Combine keywords from all rows with the same ID
            combined_keywords = '; '.join(str(keyword) for keyword in group[keyword_column] if pd.notna(keyword) and str(keyword).strip())
            # Keep the first row but update the keyword column with combined values
            first_row = group.iloc[0].copy()
            first_row[keyword_column] = combined_keywords
            return first_row
        else:
            return group.iloc[0]
    
    # Apply the combination logic
    df_clean = df.groupby(id_column).apply(combine_keywords).reset_index(drop=True)
    return df_clean

def get_predefined_keywords():
    """Return predefined keywords for filtering"""
    return [
        "miroiterie",
        "métallerie",
        "menuiserie extérieure",
        "Travaux de menuiserie et de charpenterie",
        "Pose de portes et de fenêtres et d'éléments accessoires",
        "Pose d'encadrements de portes et de fenêtres",
        "Pose d'encadrements de portes",
        "Pose d'encadrements de fenêtres",
        "Pose de seuils",
        "Poses de portes et de fenêtres",
        "Pose de portes",
        "Pose de fenêtres",
        "Pose de menuiseries métalliques, excepté portes et fenêtres",
        "Travaux de cloisonnement",
        "Installation de volets",
        "Travaux d'installation de stores",
        "Travaux d'installation de vélums",
        "Travaux d'installation de volets roulants",
        "Serrurerie",
        "Services de serrurerie",
        "Menuiserie pour la construction",
        "Travaux de menuiserie",
        "Clôtures",
        "Clôtures de protection",
        "Travaux d'installation de clôtures, de garde-corps et de dispositifs de sécurité",
        "Pose de clôtures",
        "Ascenseurs, skips, monte-charges, escaliers mécaniques et trottoirs roulants",
        "Escaliers mécaniques",
        "Pièces pour ascenseurs, skips ou escaliers mécaniques",
        "Pièces pour escaliers mécaniques",
        "Escaliers",
        "Escaliers pliants",
        "Travaux d'installation d'ascenseurs et d'escaliers mécaniques",
        "Travaux d'installation d'escaliers mécaniques",
        "Services de réparation et d'entretien d'escaliers mécaniques",
        "Services d'installation de matériel de levage et de manutention, excepté ascenseurs et escaliers mécaniques",
        "45420000", "45421100", "45421110", "45421111", "45421112", "45421120", 
        "45421130", "45421131", "45421132", "45421140", "45421141", "45421142", 
        "45421143", "45421144", "45421145", "44316500", "98395000", "44220000", 
        "45421000", "34928200", "34928310", "45340000", "45342000", "42416000", 
        "42416400", "42419500", "42419530", "44233000", "44423220", "45313000", 
        "45313200", "50740000", "51511000",
    ]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main page"""
    predefined_keywords = get_predefined_keywords()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "predefined_keywords": predefined_keywords,
        "today": date.today().isoformat()
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/process")
async def process_data(
    target_date: str = Form(...),
    selected_keywords: List[str] = Form(...),
    custom_keywords: str = Form("")
):
    """Start the data processing"""
    process_id = f"process_{int(time.time())}"
    
    # Combine keywords
    all_keywords = selected_keywords.copy()
    if custom_keywords:
        custom_keywords_list = [k.strip() for k in custom_keywords.split('\n') if k.strip()]
        all_keywords.extend(custom_keywords_list)
    
    if not all_keywords:
        raise HTTPException(status_code=400, detail="Please select at least one keyword")
    
    # Initialize processing state
    processing_state[process_id] = {
        'status': 'starting',
        'current_step': 'initializing',
        'total_records': 0,
        'processed_records': 0,
        'current_record': '',
        'result': None,
        'keywords': all_keywords,
        'target_date': target_date
    }
    
    # Run processing in background
    asyncio.create_task(run_processing(process_id, target_date, all_keywords))
    
    return JSONResponse({"process_id": process_id, "status": "started"})

async def run_processing(process_id: str, target_date: str, all_keywords: List[str]):
    """Run the full processing in background"""
    try:
        MAX_RECORDS = 10000
        
        # Step 1: Extract data
        processing_state[process_id]['current_step'] = 'data_extraction'
        all_records = get_all_records_for_date(target_date, MAX_RECORDS)
        
        if not all_records:
            processing_state[process_id]['status'] = 'error'
            processing_state[process_id]['error'] = f"No records found for date {target_date}"
            return
        
        processing_state[process_id]['total_records'] = len(all_records)
        
        # Create DataFrame
        df = create_excel_simple(all_records, target_date)
        
        # Step 2: Filter by keywords
        processing_state[process_id]['current_step'] = 'keyword_filtering'
        filtered_df = filter_by_keywords(df, all_keywords)
        
        if filtered_df.empty:
            processing_state[process_id]['status'] = 'error'
            processing_state[process_id]['error'] = "No matches found for the selected keywords"
            return
        
        # Step 3: Remove duplicates
        processing_state[process_id]['current_step'] = 'deduplication'
        available_columns = filtered_df.columns.tolist()
        id_column = available_columns[0]
        keyword_column = available_columns[-1]
        df_clean = remove_duplicates(filtered_df, id_column, keyword_column)
        
        # Step 4: Process PDFs
        processing_state[process_id]['current_step'] = 'pdf_processing'
        processed_df = extract_pdf_content(df_clean, process_id)
        
        # Create summary table
        summary_table = pd.DataFrame({
            "Keywords": processed_df.get('keyword', 'N/A'),
            'Acheteur': processed_df.get('nomacheteur', 'N/A'),
            'Objet': processed_df.get('objet', 'N/A'),
            'Lots': processed_df.get('lot_numbers', ''),
            'Visite Obligatoire': processed_df.get('visite_obligatoire', 'no'),
            'Département': processed_df.get('code_departement', 'N/A'),
            'Date Limite': processed_df.get('datelimitereponse', 'Pas Mentionné'),
            'PDF Link': processed_df.get('generated_link', 'N/A')
        })
        
        processing_state[process_id]['summary_table'] = summary_table.to_dict('records')
        processing_state[process_id]['status'] = 'completed'
        
    except Exception as e:
        processing_state[process_id]['status'] = 'error'
        processing_state[process_id]['error'] = str(e)

@app.get("/progress/{process_id}")
async def get_progress(process_id: str):
    """Get processing progress"""
    if process_id not in processing_state:
        raise HTTPException(status_code=404, detail="Process not found")
    
    return JSONResponse(processing_state[process_id])

@app.get("/download/{process_id}")
async def download_results(process_id: str):
    """Download results as Excel file"""
    if process_id not in processing_state:
        raise HTTPException(status_code=404, detail="Process not found")
    
    if processing_state[process_id]['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Process not completed")
    
    # Convert result back to DataFrame
    result_data = processing_state[process_id]['result']
    df = pd.DataFrame(result_data)
    
    # Create Excel file in memory
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, engine='openpyxl')
    excel_buffer.seek(0)
    
    filename = f"BOAMP_Full_Results_{processing_state[process_id]['target_date']}_{datetime.now().strftime('%H%M%S')}.xlsx"
    
    return StreamingResponse(
        excel_buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/download-summary/{process_id}")
async def download_summary(process_id: str):
    """Download summary table as CSV"""
    if process_id not in processing_state:
        raise HTTPException(status_code=404, detail="Process not found")
    
    if processing_state[process_id]['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Process not completed")
    
    # Get summary table
    summary_data = processing_state[process_id]['summary_table']
    df = pd.DataFrame(summary_data)
    
    # Create CSV in memory
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    csv_buffer.seek(0)
    
    filename = f"BOAMP_Summary_{processing_state[process_id]['target_date']}_{datetime.now().strftime('%H%M%S')}.csv"
    
    return StreamingResponse(
        io.BytesIO(csv_buffer.getvalue().encode('utf-8-sig')),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)