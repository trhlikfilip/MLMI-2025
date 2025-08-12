import os
import pandas as pd
from transformers import AutoConfig
from bias_helper import cs_generative, cs_discriminative, ss_generative, ss_discriminative
import torch

code_root = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/test5/code/Users/filip.trhlik/StereoSet'
dev_file = os.path.join(code_root, 'data', 'dev.json')

def model_type(model_name: str) -> bool:
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if getattr(cfg, "is_decoder", False) or getattr(cfg, "is_encoder_decoder", False):
        return True
    for arch in getattr(cfg, "architectures", []) or []:
        if any(tag in arch for tag in ["LMHeadModel", "CausalLM", "ForConditionalGeneration"]):
            return True
    return False

def merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame, suffixes=('_ss', '_cp')) -> pd.DataFrame:
    if df1.empty and df2.empty:
        return pd.DataFrame()
    if df1.empty:
        return df2.copy()
    if df2.empty:
        return df1.copy()
    common = df1.columns.intersection(df2.columns).tolist()
    if not common:
        df1_renamed = df1.add_suffix(suffixes[0])
        df2_renamed = df2.add_suffix(suffixes[1])
        return pd.concat([df1_renamed, df2_renamed], axis=1)
    unique1 = df1.columns.difference(df2.columns).tolist()
    unique2 = df2.columns.difference(df1.columns).tolist()
    df1 = df1[common + unique1]
    df2 = df2[common + unique2]
    return pd.merge(df1, df2, on=common, how='outer', suffixes=suffixes)

def evaluate_and_combine(models: list[str]) -> pd.DataFrame:
    gen_models = [m for m in models if model_type(m)]
    dis_models = [m for m in models if not model_type(m)]
    ss_dis_df = ss_discriminative(dis_models, dev_file, code_root) if dis_models else pd.DataFrame()
    torch.cuda.empty_cache()
    cp_dis_df = cs_discriminative(dis_models) if dis_models else pd.DataFrame()
    torch.cuda.empty_cache()
    ss_gen_df = ss_generative(gen_models, dev_file, code_root) if gen_models else pd.DataFrame()
    torch.cuda.empty_cache()
    cp_gen_df = cs_generative(gen_models) if gen_models else pd.DataFrame()
    merged_dis = merge_datasets(ss_dis_df, cp_dis_df, suffixes=('_ss', '_cp')) if not ss_dis_df.empty or not cp_dis_df.empty else pd.DataFrame()
    merged_gen = merge_datasets(ss_gen_df, cp_gen_df, suffixes=('_ss', '_cp')) if not ss_gen_df.empty or not cp_gen_df.empty else pd.DataFrame()
    if not merged_dis.empty:
        merged_dis['model_type'] = 'discriminative'
    if not merged_gen.empty:
        merged_gen['model_type'] = 'generative'
    combined = [df for df in [merged_dis, merged_gen] if not df.empty]
    return pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()

if __name__ == "__main__":
    models = [
        "ltg/ltg-bert-babylm"
    ]
    combined_df = evaluate_and_combine(models)
    combined_df.to_csv("ltg_bert.csv", index=False)
    print(combined_df.shape)
