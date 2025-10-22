

def compute_accuracy(y_pred_list, y_test_list):
    acc, false_positive, false_negative = 0, 0, 0
    for y_pred, y_test in zip(y_pred_list, y_test_list):
        if (y_pred >= 0.5).int() == y_test:

            acc += 1
        else:
            if y_test:
                false_positive += 1
            else:
                false_negative += 1

    n_positive = sum(y_test_list)
    acc = acc / len(y_test_list)
    fp_accu = false_positive / n_positive
    fn_accu = false_negative / (len(y_test_list) - n_positive)
    return acc, fp_accu, fn_accu
