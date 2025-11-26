
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