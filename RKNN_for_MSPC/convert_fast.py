#!/usr/bin/env python3
# coding: utf-8

import datetime
import os
from rknn.api import RKNN
from sys import exit

def convert_model(onnx_model, target, output_dir, dataset, quantize, hybrid_quantize, quantize_analysis, output_names=None, trim_outputs=False):
    base_name = os.path.splitext(os.path.basename(onnx_model))[0]
    rknn_model = os.path.join(output_dir, f"{base_name}_{target}.rknn")

    print(f"\n=== Converting model for {target} ===")

    rknn = RKNN(verbose=True)
    timedate_iso = datetime.datetime.now().isoformat()

    print('--> Config model')
    rknn.config(
        target_platform=target,
        optimization_level=3,

        # ★ 修复 fold_constant 报错
        disable_rules=['FOLD_CONSTANT_OP'],

        quantized_algorithm='kl_divergence',
        quantized_method='channel',
        model_pruning=True,
        custom_string=f"converted {timedate_iso}",
    )
    print('done')

    print('--> Loading model')
    effective_outputs = output_names if trim_outputs else None
    ret = rknn.load_onnx(model=onnx_model, outputs=effective_outputs)
    if ret != 0:
        print('Load model failed!')
        return ret
    print('done')

    if hybrid_quantize:
        if not output_names:
            print('Hybrid quantization requires detection head outputs.')
            return -1

        subgraph_end = "output0"
        subgraphs = [[s, subgraph_end] for s in output_names]

        print(f"--> Hybrid quantization step 1: {subgraphs}")
        ret = rknn.hybrid_quantization_step1(dataset=dataset,
                                             custom_hybrid=subgraphs)
        if ret != 0:
            print('Hybrid quantization step 1 failed!')
            return ret
        print('done')

        rknn.release()
        rknn = RKNN(verbose=True)

        print(f"--> Hybrid quantization step 2")
        rknn.hybrid_quantization_step2(
            model_input=onnx_model.replace(".onnx", ".model"),
            data_input=onnx_model.replace(".onnx", ".data"),
            model_quantization_cfg=onnx_model.replace(".onnx", ".quantization.cfg")
        )
        print('done')

        print(f"--> Export RKNN model: {rknn_model}")
        rknn.export_rknn(rknn_model)
        print('done')
        return 0

    print('--> Building model')
    ret = rknn.build(do_quantization=quantize, dataset=dataset)
    if ret != 0:
        print('Build model failed!')
        return ret
    print('done')

    print('--> Export RKNN model')
    ret = rknn.export_rknn(rknn_model)
    print('done')

    return 0


def main():
    onnx_model = './YOLOv11-M-s320.onnx'
    output_dir = '.'
    dataset = './dataset.txt'
    quantize = False
    hybrid_quantize = True
    quantize_analysis = False

    head_outputs = [
        '/model.23/cv2.0/cv2.0.2/Conv_output_0',
        '/model.23/cv3.0/cv3.0.2/Conv_output_0',
        '/model.23/cv2.1/cv2.1.2/Conv_output_0',
        '/model.23/cv3.1/cv3.1.2/Conv_output_0',
        '/model.23/cv2.2/cv2.2.2/Conv_output_0',
        '/model.23/cv3.2/cv3.2.2/Conv_output_0'
    ]

    trim_outputs = False
    target = 'rk3576'

    os.makedirs(output_dir, exist_ok=True)
    print(f'\n=== Converting model for {target} ===')

    ret = convert_model(
        onnx_model, target, output_dir,
        dataset, quantize, hybrid_quantize,
        quantize_analysis, head_outputs, trim_outputs
    )

    if ret != 0:
        exit(ret)

    print('\nConversion completed successfully.')

if __name__ == '__main__':
    main()
