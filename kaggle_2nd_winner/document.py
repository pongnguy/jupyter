#from collections import namedtuple
import collections

DocSpan = collections.namedtuple("DocSpan", ["start", "length"])


def get_spans(doc_stride, max_tokens_for_doc, max_len):
    doc_spans = []
    start_offset = 0
    while start_offset < max_len:
        length = max_len - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(DocSpan(start=start_offset, length=length))
        if start_offset + length == max_len:
            break
        start_offset += min(length, doc_stride)
    return doc_spans


def convert_examples_to_crops(examples_gen, tokenizer, max_seq_length,
                              doc_stride, max_query_length, is_training,
                              cls_token='[CLS]', sep_token='[SEP]', pad_id=0,
                              sequence_a_segment_id=0,
                              sequence_b_segment_id=1,
                              cls_token_segment_id=0,
                              pad_token_segment_id=0,
                              mask_padding_with_zero=True,
                              p_keep_impossible=None,
                              sep_token_extra=False):
    """Loads a data file into a list of `InputBatch`s."""
    assert p_keep_impossible is not None, '`p_keep_impossible` is required'
    unique_id = 1000000000
    num_short_pos, num_short_neg = 0, 0
    num_long_pos, num_long_neg = 0, 0
    sub_token_cache = {}
    # max_N, max_M = 1024, 1024
    # f = np.zeros((max_N, max_M), dtype=np.float32)

    crops = []
    for example_index, example in enumerate(examples_gen):
        if example_index % 1000 == 0 and example_index > 0:
            logger.info('Converting %s: short_pos %s short_neg %s'
                ' long_pos %s long_neg %s',
                example_index, num_short_pos, num_short_neg,
                num_long_pos, num_long_neg)

        query_tokens = tokenizer.tokenize(example.question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        # this takes the longest!
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []

        for i, token in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = sub_token_cache.get(token)
            if sub_tokens is None:
                sub_tokens = tokenizer.tokenize(token)
                sub_token_cache[token] = sub_tokens
            tok_to_orig_index.extend([i for _ in range(len(sub_tokens))])
            all_doc_tokens.extend(sub_tokens)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.short_is_impossible:
            tok_start_position = -1
            tok_end_position = -1

        if is_training and not example.short_is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[
                    example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

        tok_long_position = None
        if is_training and example.long_is_impossible:
            tok_long_position = -1

        if is_training and not example.long_is_impossible:
            tok_long_position = orig_to_tok_index[example.long_position]

        # For Bert: [CLS] question [SEP] paragraph [SEP]
        special_tokens_count = 3
        if sep_token_extra:
            # For Roberta: <s> question </s> </s> paragraph </s>
            special_tokens_count += 1
        max_tokens_for_doc = max_seq_length - len(query_tokens) - special_tokens_count
        assert max_tokens_for_doc > 0
        # We can have documents that are longer than the maximum
        # sequence length. To deal with this we do a sliding window
        # approach, where we take chunks of the up to our max length
        # with a stride of `doc_stride`.
        doc_spans = get_spans(doc_stride, max_tokens_for_doc, len(all_doc_tokens))
        for doc_span_index, doc_span in enumerate(doc_spans):
            # Tokens are constructed as: CLS Query SEP Paragraph SEP
            tokens = []
            token_to_orig_map = UNMAPPED * np.ones((max_seq_length, ), dtype=np.int32)
            token_is_max_context = np.zeros((max_seq_length, ), dtype=np.bool)
            token_type_ids = []

            # p_mask: mask with 1 for token than cannot be in the
            # answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token
            # (set to 0) (not sure why...)
            # p_mask = []

            short_is_impossible = example.short_is_impossible
            start_position = None
            end_position = None
            special_tokens_offset = special_tokens_count - 1
            doc_offset = len(query_tokens) + special_tokens_offset
            if is_training and not short_is_impossible:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    start_position = 0
                    end_position = 0
                    short_is_impossible = True
                else:
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            long_is_impossible = example.long_is_impossible
            long_position = None
            if is_training and not long_is_impossible:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                # out of span
                if not (tok_long_position >= doc_start and tok_long_position <= doc_end):
                    long_position = 0
                    long_is_impossible = True
                else:
                    long_position = tok_long_position - doc_start + doc_offset

            # drop impossible samples
            if long_is_impossible:
                if np.random.rand() > p_keep_impossible:
                    continue

            # CLS token at the beginning
            tokens.append(cls_token)
            token_type_ids.append(cls_token_segment_id)
            # p_mask.append(0)  # can be answer

            # Query
            tokens += query_tokens
            token_type_ids += [sequence_a_segment_id] * len(query_tokens)
            # p_mask += [1] * len(query_tokens)  # can not be answer

            # SEP token
            tokens.append(sep_token)
            token_type_ids.append(sequence_a_segment_id)
            # p_mask.append(1)  # can not be answer
            if sep_token_extra:
                tokens.append(sep_token)
                token_type_ids.append(sequence_a_segment_id)
                # p_mask.append(1)

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                # We add `example.crop_start` as the original document
                # is already shifted
                token_to_orig_map[len(tokens)] = tok_to_orig_index[
                    split_token_index] + example.crop_start

                token_is_max_context[len(tokens)] = check_is_max_context(doc_spans,
                    doc_span_index, split_token_index)
                tokens.append(all_doc_tokens[split_token_index])
                token_type_ids.append(sequence_b_segment_id)
                # p_mask.append(0)  # can be answer

            paragraph_len = doc_span.length

            # SEP token
            tokens.append(sep_token)
            token_type_ids.append(sequence_b_segment_id)
            # p_mask.append(1)  # can not be answer

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_id)
                attention_mask.append(0 if mask_padding_with_zero else 1)
                token_type_ids.append(pad_token_segment_id)
                # p_mask.append(1)  # can not be answer

            # reduce memory, only input_ids needs more bits
            input_ids = np.array(input_ids, dtype=np.int32)
            attention_mask = np.array(attention_mask, dtype=np.bool)
            token_type_ids = np.array(token_type_ids, dtype=np.uint8)
            # p_mask = np.array(p_mask, dtype=np.bool)

            if is_training and short_is_impossible:
                start_position = CLS_INDEX
                end_position = CLS_INDEX

            if is_training and long_is_impossible:
                long_position = CLS_INDEX

            if example_index in (0, 10):
                # too spammy otherwise
                if doc_span_index in (0, 5):
                    logger.info("*** Example ***")
                    logger.info("unique_id: %s" % (unique_id))
                    logger.info("example_index: %s" % (example_index))
                    logger.info("doc_span_index: %s" % (doc_span_index))
                    logger.info("tokens: %s" % " ".join(tokens))
                    # logger.info("token_to_orig_map: %s" % " ".join([
                    #     "%d:%d" % (x, y) for (x, y) in enumerate(token_to_orig_map)]))
                    # logger.info("token_is_max_context: %s" % " ".join([
                    #     "%d:%s" % (x, y) for (x, y) in enumerate(token_is_max_context)
                    # ]))
                    logger.info("input_ids: %s" % input_ids)
                    logger.info("attention_mask: %s" % np.uint8(attention_mask))
                    logger.info("token_type_ids: %s" % token_type_ids)
                    if is_training and short_is_impossible:
                        logger.info("short impossible example")
                    if is_training and long_is_impossible:
                        logger.info("long impossible example")
                    if is_training and not short_is_impossible:
                        answer_text = " ".join(tokens[start_position: end_position + 1])
                        logger.info("start_position: %d" % (start_position))
                        logger.info("end_position: %d" % (end_position))
                        logger.info("answer: %s" % (answer_text))

            if short_is_impossible:
                num_short_neg += 1
            else:
                num_short_pos += 1

            if long_is_impossible:
                num_long_neg += 1
            else:
                num_long_pos += 1

            crop = Crop(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                # p_mask=p_mask,
                paragraph_len=paragraph_len,
                start_position=start_position,
                end_position=end_position,
                long_position=long_position,
                short_is_impossible=short_is_impossible,
                long_is_impossible=long_is_impossible)
            crops.append(crop)
            unique_id += 1

    return crops


def check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index