import config


def get_detection_report(predictions):

    predictions = predictions.tolist()
    predictions = [[round(elem * 100, 2) for elem in sublist] for sublist in predictions]
    predictions = [[float(elem) for elem in sublist] for sublist in predictions]
    predictions = predictions[0]
    result = dict(zip(config.CLASSES, predictions))
    result = sorted(result.items(), key=lambda x: x[1], reverse=True)
    final_result_report = dict(result)

    return final_result_report
