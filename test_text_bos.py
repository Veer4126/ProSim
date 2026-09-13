"""The BOS-strip fix in text_attns.LlamaTextAttn._get_llm_text_emd, without Llama weights.

    python3 test_text_bos.py        (prosim_v4.sif; needs only the tokenizer files)

Calls the REAL _config_tokenizer and _get_llm_text_emd on a stub whose only
fake part is the embedding table: it returns the token ids as floats, so the
ids the LLM would embed can be read straight off the output. The ORIGINAL code
is replayed by applying its unconditional [:, 1:] to the same tokenizer output.

Control: a tokenizer forced to prepend BOS. There the fix must strip exactly
that token and produce the same ids as the BOS-free tokenizer.
"""
import copy
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, "/scratch/veerk41/ProSim")
from prosim.models.condition_transformer.text_attns import LlamaTextAttn

TOK_DIR = "/scratch/veerk41/ProSim/Meta-Llama-3-8B-Instruct-HF"
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


class IdsAsEmbedding(torch.nn.Module):
    def forward(self, ids):
        return ids.float().unsqueeze(-1)


class StubLLM:
    def get_input_embeddings(self):
        return IdsAsEmbedding()

    def resize_token_embeddings(self, n):
        pass


def make_attn(tokenizer):
    attn = LlamaTextAttn.__new__(LlamaTextAttn)          # skip __init__: no weights
    torch.nn.Module.__init__(attn)
    attn.llm_tokenizer = tokenizer
    attn.llm_model = StubLLM()
    attn.max_txt_len = 128
    LlamaTextAttn._config_tokenizer(attn)                 # the real tokenizer setup
    return attn


def fixed_ids(attn, texts):
    out = LlamaTextAttn._get_llm_text_emd(attn, texts, "cpu")
    ids = out["llm_emd"].squeeze(-1).long()
    assert torch.equal(ids, out["input_ids"]), "stub embedding must return the ids"
    assert out["attn_mask"].shape == ids.shape, "mask and ids must stay aligned"
    return ids, out["attn_mask"]


def original_ids(attn, texts):
    t = attn.llm_tokenizer(texts, return_tensors="pt", padding="longest",
                           truncation=True, max_length=attn.max_txt_len)
    return t.input_ids[:, 1:], t.attention_mask[:, 1:]


def words(tok, ids, mask):
    return [tok.decode(r[m.bool()]) for r, m in zip(ids, mask)]


def main():
    texts = ["The <A0> turns left at the intersection.",
             "<A0> slows down and stops.",
             "Go straight, <A1>."]

    print("\n=== 1. the real Llama-3 tokenizer, as ProSim configures it ===")
    attn = make_attn(AutoTokenizer.from_pretrained(TOK_DIR))
    tok = attn.llm_tokenizer
    raw = tok(texts, return_tensors="pt", padding="longest").input_ids
    check("this tokenizer adds no BOS", not (raw[:, 0] == tok.bos_token_id).any(),
          f"first ids {raw[:, 0].tolist()}, bos {tok.bos_token_id}")
    f_ids, f_mask = fixed_ids(attn, texts)
    o_ids, o_mask = original_ids(attn, texts)
    fw, ow = words(tok, f_ids, f_mask), words(tok, o_ids, o_mask)
    for t, a, b in zip(texts, fw, ow):
        print(f"    input    {t!r}\n    fixed    {a!r}\n    original {b!r}")
    check("fixed: the LLM sees every input word", fw == texts, str(fw))
    check("original: the first word is deleted",
          ow[0].strip() == texts[0].split(" ", 1)[1] and not ow[0].strip().startswith("The"),
          repr(ow[0]))
    a0 = tok.convert_tokens_to_ids("<A0>")
    check("original: a prompt STARTING with <A0> loses the agent token itself",
          (o_ids[1] == a0).sum().item() == 0 and (f_ids[1] == a0).sum().item() == 1,
          f"<A0> id {a0}: original count {(o_ids[1] == a0).sum().item()}, "
          f"fixed count {(f_ids[1] == a0).sum().item()}")

    print("\n=== 2. CONTROL: a tokenizer that DOES prepend BOS ===")
    bos_tok = copy.deepcopy(AutoTokenizer.from_pretrained(TOK_DIR))
    from tokenizers.processors import TemplateProcessing
    bos_tok._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos_tok.bos_token} $A", special_tokens=[(bos_tok.bos_token, bos_tok.bos_token_id)])
    battn = make_attn(bos_tok)
    braw = battn.llm_tokenizer(texts, return_tensors="pt", padding="longest").input_ids
    check("control tokenizer really prepends BOS", (braw[:, 0] == bos_tok.bos_token_id).all().item(),
          f"first ids {braw[:, 0].tolist()}")
    b_ids, b_mask = fixed_ids(battn, texts)
    check("fix strips exactly the BOS: same ids as the BOS-free tokenizer",
          torch.equal(b_ids, f_ids) and torch.equal(b_mask, f_mask),
          str(words(bos_tok, b_ids, b_mask)))
    bo_ids, bo_mask = original_ids(battn, texts)
    check("and there the fix agrees with the original", torch.equal(bo_ids, b_ids))

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
