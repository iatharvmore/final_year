import pandas as pd

def generate_synthetic_data():
    # HRMS Data (Human Resources Management System)
    hrms_data = pd.DataFrame({
        'EmployeeID': ['EMP001', 'EMP002', 'EMP003', 'EMP004', 'EMP005'],
        'Name': ['Alice Smith', 'Bob Johnson', 'Charlie Brown', 'Diana Prince', 'Evan Wright'],
        'Department': ['Sales', 'Engineering', 'Marketing', 'Sales', 'Engineering'],
        'Role': ['Account Executive', 'Senior Developer', 'Marketing Specialist', 'Sales Manager', 'QA Engineer'],
        'AttendanceRate_Pct': [98, 92, 85, 99, 95],
        'LeaveBalance_Days': [12, 5, 2, 15, 8],
        'PerformanceScore': [4.5, 3.8, 2.9, 4.8, 4.0]
    })
    
    # ERP Data (Enterprise Resource Planning)
    erp_data = pd.DataFrame({
        'EmployeeID': ['EMP001', 'EMP002', 'EMP003', 'EMP004', 'EMP005'],
        'CurrentProject': ['Project Alpha', 'Platform Migration', 'Q3 Campaign', 'Project Alpha', 'Platform Migration'],
        'BillableHours_MTD': [150, 160, 120, 155, 140],
        'BudgetUtilization_Pct': [85, 95, 110, 80, 75],
        'TrainingHours_YTD': [10, 40, 5, 20, 25]
    })
    
    # CRM Data (Customer Relationship Management)
    crm_data = pd.DataFrame({
        'EmployeeID': ['EMP001', 'EMP002', 'EMP003', 'EMP004', 'EMP005'],
        'DealsClosed_YTD': [12, 0, 0, 15, 0],
        'SalesTarget_Pct': [110, pd.NA, pd.NA, 130, pd.NA],
        'RevenueGenerated_USD': [120000, 0, 0, 150000, 0],
        'ClientSatisfactionScore': [4.8, pd.NA, pd.NA, 4.9, 4.2]
    })
    
    # Merge for a holistic view
    merged_data = pd.merge(hrms_data, erp_data, on='EmployeeID')
    merged_data = pd.merge(merged_data, crm_data, on='EmployeeID')
    
    return hrms_data, erp_data, crm_data, merged_data
