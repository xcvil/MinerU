import os
import fitz
import base64
import litserve as ls
from uuid import uuid4
from fastapi import HTTPException
from filetype import guess_extension
import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='LitServer Configuration Parser')
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='part_0',
        help='Directory path for output data'
    )
    
    parser.add_argument(
        '--accelerator',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu', 'tpu'],
        help='Type of accelerator to use'
    )
    
    parser.add_argument(
        '--devices',
        type=int,
        default=2,
        help='Number/list of devices to use'
    )
    
    parser.add_argument(
        '--workers-per-device',
        type=int,
        default=4,
        help='Number of workers per device'
    )
    
    parser.add_argument(
        '--timeout',
        action='store_true',
        default=False,
        help='Enable timeout (default: False)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port number for the server'
    )
    
    return parser


class MinerUAPI(ls.LitAPI):
    def __init__(self, output_dir='/tmp'):
        self.output_dir = output_dir

    def setup(self, device):
        if device.startswith('cuda'):
            os.environ['CUDA_VISIBLE_DEVICES'] = device.split(':')[-1]
            print(f'Initializing model on device {device}...')
            import torch
            if torch.cuda.device_count() > 1:
                raise RuntimeError("Remove any CUDA actions before setting 'CUDA_VISIBLE_DEVICES'.")
        
        import torch
        from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton
        model_manager = ModelSingleton()
        model_manager.get_model(True, False)
        model_manager.get_model(False, False)
        print(f'Model initialization complete on {device}!')

    def decode_request(self, request):
        file = request['file']
        file = self.to_pdf(file)
        opts = request.get('kwargs', {})
        opts.setdefault('debug_able', False)
        opts.setdefault('parse_method', 'auto')
        return file, opts

    def predict(self, inputs):
        from magic_pdf.tools.common import do_parse
        try:
            do_parse(self.output_dir, pdf_name := str(uuid4()), inputs[0], [], **inputs[1])
            return pdf_name
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            self.clean_memory()

    def encode_response(self, response):
        return {'output_dir': response}

    def clean_memory(self):
        import gc
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def to_pdf(self, file_base64):
        try:
            file_bytes = base64.b64decode(file_base64)
            file_ext = guess_extension(file_bytes)
            with fitz.open(stream=file_bytes, filetype=file_ext) as f:
                if f.is_pdf: return f.tobytes()
                return f.convert_to_pdf()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    print("All Arguments for LitServer Configuration:", vars(args))

    # server = ls.LitServer(
    #     MinerUAPI(output_dir='/home/zhengx46/attritiondata/part_0'),
    #     accelerator='cuda',
    #     devices=2,
    #     workers_per_device=4,
    #     timeout=False
    # )
    # server.run(port=8000)

    server = ls.LitServer(
        MinerUAPI(output_dir=args.output_dir),
        accelerator=args.accelerator,
        devices=args.devices,
        workers_per_device=args.workers_per_device,
        timeout=args.timeout,
        track_requests=True
    )
    server.run(port=args.port)
