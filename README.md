graph TD;
  input --> retrieve_grade_reports 

  retrieve_grade_reports --> process_grades_columns
  
  process_grades_columns --> df_to_model;