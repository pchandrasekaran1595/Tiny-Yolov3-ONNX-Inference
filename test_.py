import pytest
from main import INPUT_PATH, get_image, Model


@pytest.mark.parametrize(
    "filename, expected_label, expected_score, expected_box", 
    [("Test_1.jpg", "car", 0.98, (662, 418, 1674, 836)),
    ("Test_2.jpg", "car", 0.79, (297, 436, 1571, 964))]
)
def test_detection(filename, expected_label, expected_score, expected_box):
    model = Model()
    model.setup()

    image = get_image(INPUT_PATH + "/" + filename)
    label, score, box = model.infer(image)

    assert label == expected_label
    assert score >= expected_score
    assert box == expected_box
