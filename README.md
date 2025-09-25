# GPT-2 Architecture (Tensor Mapping)

```mermaid
flowchart TD

    subgraph Embeddings
        WTE["transformer.wte.weight\n(Token Embeddings)"]
        WPE["transformer.wpe.weight\n(Positional Embeddings)"]
    end

    WTE --> B0
    WPE --> B0

    subgraph Block0["Transformer Block 0 (h.0)"]
        LN1_0["ln_1.weight / ln_1.bias"]
        ATTQKV_0["attn.c_attn.weight / bias\n(Q, K, V projections)"]
        ATTOUT_0["attn.c_proj.weight / bias\n(Attention Output)"]
        LN2_0["ln_2.weight / ln_2.bias"]
        MLPFC_0["mlp.c_fc.weight / bias\n(Feedforward Expand)"]
        MLPPROJ_0["mlp.c_proj.weight / bias\n(Feedforward Project)"]
    end

    B0[LN1_0 + Attention + LN2_0 + MLP] --> B1

    subgraph Block1["Transformer Block 1 (h.1)"]
        LN1_1["ln_1.weight / ln_1.bias"]
        ATTQKV_1["attn.c_attn.weight / bias"]
        ATTOUT_1["attn.c_proj.weight / bias"]
        LN2_1["ln_2.weight / ln_2.bias"]
        MLPFC_1["mlp.c_fc.weight / bias"]
        MLPPROJ_1["mlp.c_proj.weight / bias"]
    end

    B1 --> Block1

    %% Repeat ... (Blocks 2–11 for GPT-2 small)

    Block1 --> LNFinal

    subgraph FinalLayerNorm
        LNFinal["transformer.ln_f.weight / bias"]
    end

    LNFinal --> LMHead

    subgraph LMHead
        LMHead["lm_head.weight (tied to wte.weight)"]
    end
```
