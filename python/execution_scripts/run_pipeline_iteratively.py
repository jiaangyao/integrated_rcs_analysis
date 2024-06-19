import subprocess
import pandas as pd
import runpy
import sys

# CSV_PATH = 'path/to/your/csv/file.csv'
# PYTHON_PATH = 'path/to/your/python/executable' 

CSV_PATH = '/home/claysmyth/code/configs/csvs/pipeline_run_settings.csv'
# CSV_PATH = '/home/claysmyth/code/configs/csvs/pipeline_run_settings_by_region.csv'
# CSV_PATH = '/home/claysmyth/code/configs/csvs/pipeline_run_settings_6_second_epochs.csv'
# CSV_PATH = '/home/claysmyth/code/configs/csvs/pipeline_run_settings_03_debug.csv'
# CSV_PATH = '/home/claysmyth/code/configs/csvs/BG_reruns.csv'
# CSV_PATH = '/home/claysmyth/code/configs/csvs/pipeline_run_settings_by_FTG.csv'
# CSV_PATH = '/home/claysmyth/code/configs/csvs/pipeline_run_training_debug.csv'

def execute_commands_from_csv(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    for index, row in df.iterrows():
        
        if row['run_row'] == False: # Skip row if the 'run_row' column is False, allowing user to not execute certain rows of CSV
            continue
        
        # Start building the command
        # command = [f"{PYTHON_PATH}", "python/pipeline/pipeline_main.py"]
        command = ["/home/claysmyth/code/integrated_rcs_analysis/python/pipeline/pipeline_main.py"]
        
        # Iterate over each column and add to the command if the value exists
        for column in df.columns:
            
            if column == 'run_row': # Ignore the 'run_row' column, because we already checked it
                continue
            
            # Assuming 'flag' column should be treated differently
            if column == 'flags':
                if pd.notna(row[column]):
                    command.append(f'--{row[column]}')  # Append flag as is
            else:
                # For other columns, only add argument if it exists (is not NaN and not empty)
                if (pd.notna(row[column]) and row[column] != '') and column != 'overrides' and column != 'additional_tags':
                    command.append(f"{column}={row[column]}")
                elif column == 'overrides':
                    # # Handle 'overrides' column by evaluating its list of arguments and adding them to the command
                    # if pd.notna(row[column]) and row[column] != '':
                    #     try:
                    #         overrides = ast.literal_eval(row[column])
                    #         if isinstance(overrides, list):
                    #             command.extend(overrides)  # Add each override arg to the command
                    #     except ValueError as e:
                    #         print(f"Error processing overrides for row {index}: {e}")
                    if pd.notna(row[column]):
                        command.extend((row[column]).split("; "))
        
        # Convert command list to string if required by subprocess.run
        command_str = ' '.join(command)
        
        # Execute the command
        print(f"Executing: {command_str}")  # Print the command to be executed (for debugging)
        # try:
        #     subprocess.run(command_str, shell=True, check=True)
        # except subprocess.CalledProcessError as e:
        #     print(f"Error executing command for row {index}")
        #     print("Moving on...")
        # print("Done!")
        
        try:
            # Hydra sometimes doens't like the way quotes are saved in CSVs, so we need to replace them
            command = [s.replace('“', '"').replace('”', '"') for s in command]
            sys.argv = command
            runpy.run_path(command[0], run_name='__main__')
        except subprocess.CalledProcessError as e:
            print(f"Error executing command for row {index}")
            print("Moving on...")
        print("Done!")
            

if __name__ == "__main__":
    execute_commands_from_csv(CSV_PATH)