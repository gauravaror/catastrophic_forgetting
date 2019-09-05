import models.utils as utils

def test_basic_catastrophic_metric():
    # This section is purely added for testing.
    test_metric_tasks = ['trec', 'cola']
    test_metric = {'trec': {'trec': 0.97, 'cola': 0.60} , 'cola': {'trec': 0.50, 'cola': 0.67}}
    test_metric_forgetting = utils.get_catastrophic_metric(test_metric_tasks, test_metric)
    assert round(test_metric_forgetting['trec'],2) == 0.38
    assert round(test_metric_forgetting['1_step'],2) == 0.38
    assert round(test_metric_forgetting['total'],2) == 0.38

    test_metric_tasks = ['subjectivity', 'sst', 'trec', 'cola']
    test_metric['subjectivity'] =  {'subjectivity': 0.87, 'sst': 0.76, 'trec': 0.64, 'cola': 0.67}
    test_metric['sst'] =  {'subjectivity': 0.21, 'sst': 0.32, 'trec': 0.26, 'cola': 0.27}
    test_metric['trec'] =  {'subjectivity': 0.18, 'sst': 0.19, 'trec': 0.76, 'cola': 0.67}
    test_metric['cola'] = {'subjectivity': 0.69, 'sst': 0.67, 'trec': 0.68, 'cola': 0.69}
    test_metric_forgetting = utils.get_catastrophic_metric(test_metric_tasks, test_metric)
    assert round(test_metric_forgetting['subjectivity'],2) == 0.23
    assert round(test_metric_forgetting['trec'],2) == 0.12
    assert round(test_metric_forgetting['sst'],2) == 0.16
    assert round(test_metric_forgetting['total'],2) == 0.50
    assert round(test_metric_forgetting['1_step'],2) == 0.09

if __name__ == "__main__":
    test_basic_catastrophic_metric()
    print("Everything passed")
