import re
import torch


def tag2triples(word_seq, tag_seq):
    assert len(word_seq)==len(tag_seq)
    triples = []
    i = 0
    while i < len(tag_seq):
        tag = tag_seq[i]
        if tag.startswith('B'):
            intent, slot = tag[2:].split('+')
            value = word_seq[i]
            j = i + 1
            while j < len(tag_seq):
                if tag_seq[j].startswith('I') and tag_seq[j][2:] == tag[2:]:
                    value += ' ' + word_seq[j]
                    i += 1
                    j += 1
                else:
                    break
            triples.append((intent, slot, value))
        i += 1
    return triples


def intent2triples(intent_seq):
    triples = []
    for intent in intent_seq:
        intent, slot, value = re.split('[+*]', intent)
        triples.append((intent, slot, value))
    return triples


def recover_intent(dataloader, intent_logits, tag_logits, tag_mask_tensor, ori_word_seq, new2ori):
    # tag_logits = [sequence_length, tag_dim]
    # intent_logits = [intent_dim]
    # tag_mask_tensor = [sequence_length]
    # new2ori = {(new_idx:old_idx),...} (after removing [CLS] and [SEP]
    max_seq_len = tag_logits.size(0)
    intents = []
    for j in range(dataloader.intent_dim):
        if intent_logits[j] > 0:
            intent, slot, value = re.split('[+*]', dataloader.id2intent[j])
            intents.append((intent, slot, value))
    tags = []
    for j in range(1 , max_seq_len -1):
        if tag_mask_tensor[j] == 1:
            value, tag_id = torch.max(tag_logits[j], dim=-1)
            tags.append(dataloader.id2tag[tag_id.item()])
    recover_tags = []
    for i, tag in enumerate(tags):
        if new2ori[i] >= len(recover_tags):
            recover_tags.append(tag)
    tag_intent = tag2triples(ori_word_seq, recover_tags)
    intents += tag_intent
    return intents
