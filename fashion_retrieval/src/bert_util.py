class InputFeatures(object):
    """
    tokens : tokens of text
    input_ids : id of tokens
    input_mask : 0/1 mask corresponding length of text
    //input_type_ids : whether the token belongs to the first or second input sequence (probably don't need this)
    """

    def __init__(self, tokens, input_ids, input_mask):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask


def convert_examples_to_features(texts, max_seq_length, tokenizer):
    """
    Input : list of texts, maximum sequence length allowed, bert tokenizer
    Output : list of features including (tokens, input_ids, input_mask)
    """

    features = []
    token_list = [tokenizer.tokenize(text) for text in texts]

    # Set maximum length
    if max_seq_length is None:
        # Account for [CLS], [SEP] -> 2 additional tokens
        max_seq_length = max(len(t) for t in texts) + 2

    # Truncate tokens to maximum allowed length
    for tokens in token_list:
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]
        input_mask = [0] + [1] * len(tokens) + [0]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # Get id of tokens and zero-pad    
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        pad_len = max_seq_length - len(input_ids)
        input_ids += [0] * pad_len
        input_mask += [0] * pad_len
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        
        features.append(
            InputFeatures(
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask))
        
    return features
