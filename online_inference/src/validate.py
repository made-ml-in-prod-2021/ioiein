from src.entities import HeartData


def check(value):
    if value is None:
        raise ValueError


def validate_data(data: HeartData) -> bool:
    try:
        check(data.id)
        check(data.age)
        check(data.chol)
        check(data.sex)
        check(data.ca)
        check(data.cp)
        check(data.trestbps)
        check(data.fbs)
        check(data.restecg)
        check(data.thalach)
        check(data.exang)
        check(data.slope)
        check(data.thal)
        check(data.oldpeak)
        return True
    except ValueError:
        return False
