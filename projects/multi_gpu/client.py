import base64
import requests
import numpy as np
from loguru import logger
from joblib import Parallel, delayed
from pathlib import Path
import pickle
import os
import pickle
import time
import socket
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process server configuration')
    parser.add_argument('--prefix', type=str, default='part_0',
                        help='Prefix for the application')
    parser.add_argument('--n-jobs', type=int, default=8,
                        help='I think it is the maximal number of threads')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Host address to run the server on')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port number to run the server on')
    
    return parser.parse_args()


def check_server_ready(host='localhost', port=8000, max_attempts=10, delay=10):
    """Check if the server is accepting connections"""
    for attempt in range(max_attempts):
        try:
            with socket.create_connection((host, port), timeout=5):
                logger.info(f"Server is ready on {host}:{port}")
                return True
        except (socket.timeout, ConnectionRefusedError):
            logger.warning(f"Server not ready (attempt {attempt + 1}/{max_attempts}), waiting {delay} seconds...")
            time.sleep(delay)
    return False


def find_pdf_files(folder_path):
    pdf_files = []
    # Convert string path to Path object
    root_dir = Path(folder_path)
    # Recursively search for PDF files
    for file_path in root_dir.rglob('*.pdf'):
        pdf_files.append(str(file_path.absolute()))
    return pdf_files


def to_b64(file_path):
    try:
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        raise Exception(f'File: {file_path} - Info: {e}')


def do_parse(file_path, host='localhost', port=8000, **kwargs):
    url = f'http://{host}:{port}/predict'

    try:
        response = requests.post(url, json={
            'file': to_b64(file_path),
            'kwargs': kwargs
        })

        if response.status_code == 200:
            output = response.json()
            output['file_path'] = file_path
            return output
        else:
            raise Exception(response.text)
    except Exception as e:
        logger.error(f'File: {file_path} - Info: {e}')


def main(prefix, njobs, host='localhost', port=8000):
    if not check_server_ready(host=host, port=port):
        logger.error("Server is not responding. Exiting.")
        return None
    
    # Get list of PDF files
    folder_path = f'/pstore/data/llm-comptox/Input/RDR_232/RDR_232_split_100/{prefix}/Processed'
    files = find_pdf_files(folder_path)
    
    # Print found PDF files
    logger.info(f"Found {len(files)} PDF files")

    n_jobs = np.clip(len(files), 1, njobs)
    results = Parallel(n_jobs, prefer='processes', verbose=10)(
        delayed(do_parse)(p, host=host, port=port) for p in files
    )

    # Create output directories if they don't exist
    save_dir = os.path.join('/home/zhengx46', 'process_data')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save results and all_files
    with open(os.path.join(save_dir, str(prefix)+'_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    logger.info(f"Saved results to {save_dir}")
    print(results)


if __name__ == '__main__':
    args = parse_args()
    print("All Arguments for Client Configuration:", vars(args))

    main(prefix=args.prefix, njobs=args.n_jobs, host=args.host, port=args.port)