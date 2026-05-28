import pytest
from magfield.frame import Frame


def test_frame_init():
    frame = Frame(size=10, step=2)
    assert frame.shape == (10, 2)
    assert frame.size == 10
    assert frame.step == 2


def test_frame_iter():
    frame = Frame(size=10, step=2)
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    frame.data = data
    iterator = iter(frame)
    assert next(iterator) == data[: frame.size]
    assert next(iterator) == data[frame.step : frame.step + frame.size]


# def test_frame_apply():
#     frame = Frame(size=10, step=2)
#     data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     def func(x):
#         return sum(x)
#     result = frame.apply(data, func)
#     assert result.shape == (5, 1)  # assuming pandas DataFrame
#     assert result.iloc[0, 0] == sum(data[:10])


def test_frame_empty_data():
    frame = Frame(size=10, step=2)
    with pytest.raises(UnboundLocalError):
        next(iter(frame))


def test_frame_invalid_size():
    frame = Frame(size=10, step=2)
    data = [1, 2, 3]
    with pytest.raises(ValueError):
        frame.apply(data, lambda x: x)
