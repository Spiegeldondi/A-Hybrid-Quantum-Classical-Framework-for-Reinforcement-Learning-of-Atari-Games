import os

def create_log_dir(output_path, id):
    if output_path is None:
        log_dir = os.path.join(os.getcwd(), "output-"+id)
    else:
        log_dir = os.path.join(output_path, "output-"+id)
    return log_dir

def create_log_files(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_files = [f for f in os.listdir(log_dir) if f.startswith("metrics") and f.endswith('.csv')]
    print("\nlen(log_files):\n", len(log_files))
    return log_files
