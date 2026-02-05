from pathlib import Path
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoModel
from bertviz import head_view, model_view
import matplotlib.pyplot as plt
import seaborn as sns


@hydra.main(version_base=None, config_path="../config", config_name="default")
def visual_analysis(config: DictConfig) -> None:
    OmegaConf.set_struct(config, True)

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    model = AutoModel.from_pretrained(config.model.model_name, output_attentions=True)

    inputs = tokenizer.encode_plus(config.settings.text_to_analyse, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"]
    token_type_ids = inputs.get("token_type_ids", None)
    attention = model(input_ids, token_type_ids=token_type_ids)[-1]
    input_id_list = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)

    # head_view, model_view
    for visualization_func in [head_view, model_view]:
        html_result = visualization_func(attention, tokens, html_action="return")
        res_path = Path(config.visualize.output_dir).joinpath(config.settings.language, visualization_func.__name__).with_suffix(".html")
        res_path.parent.mkdir(parents=True, exist_ok=True)
        with res_path.open(mode="w") as f:
            f.write(html_result.data)
 
    # heatmap
    last_attention = attention[-1]
    avg_attention = last_attention.mean(dim=0)
    if len(avg_attention.shape) > 2:
        avg_attention = avg_attention[0] if avg_attention.shape[0] == 1 else avg_attention.mean(dim=0)
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(avg_attention.cpu().detach().numpy(),
                xticklabels=tokens,
                yticklabels=tokens,
                ax=ax)
    
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=config.visualize.font_size, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=config.visualize.font_size)
    plt.title("Attention Heatmap")
    plt.xlabel("Tokens")
    plt.ylabel("Tokens")
    plt.tight_layout()

    heatmap_path = Path(config.visualize.output_dir).joinpath(config.settings.language, "heatmap.png")
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(heatmap_path, dpi=1280, bbox_inches='tight')
    plt.close()



if __name__ == "__main__":
    visual_analysis()