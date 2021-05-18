from src.entities import HeartData


def check_int(value):
    if type(value) != int:
        raise TypeError


def check_float(value):
    if type(value) != int:
        raise TypeError


def validate_data(data: HeartData) -> bool:
    try:
        check_int(data.id)
        check_int(data.age)
        check_int(data.chol)
        check_int(data.sex)
        check_int(data.ca)
        check_int(data.cp)
        check_int(data.trestbps)
        check_int(data.fbs)
        check_int(data.restecg)
        check_int(data.thalach)
        check_int(data.exang)
        check_int(data.slope)
        check_int(data.thal)
        check_float(data.oldpeak)
    except TypeError:
        return False
