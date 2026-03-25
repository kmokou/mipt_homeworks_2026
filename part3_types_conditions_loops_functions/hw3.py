from typing import Any

UNKNOWN_COMMAND_MSG = "Unknown command!"
NONPOSITIVE_VALUE_MSG = "Value must be grater than zero!"
INCORRECT_DATE_MSG = "Invalid date!"
NOT_EXISTS_CATEGORY = "Category not exists!"
OP_SUCCESS_MSG = "Added"


EXPENSE_CATEGORIES = {
    "Food": ("Supermarket", "Restaurants", "FastFood", "Coffee", "Delivery"),
    "Transport": ("Taxi", "Public transport", "Gas", "Car transport"),
    "Housing": ("Rent", "Utilities", "Repairs", "Furniture"),
    "Health": ("Pharmacy", "Doctors", "Dentist", "Lab tests"),
    "Entertainment": ("Movies", "Concerts", "Games", "Subscriptions"),
    "Clothing": ("Outerwear", "Casual", "Shoes", "Accessories"),
    "Education": ("Courses", "Books", "Tutors"),
    "Communications": ("Mobile", "Internet", "Subscriptions"),
    "Other": ("SomeCategory", "SomeOtherCategory"),
}

TRANSACTION_AMOUNT_KEY = "amount"
TRANSACTION_DATE_KEY = "date"
TRANSACTION_CATEGORY_KEY = "category"

STATS_TOTAL_CAPITAL_KEY = "total_capital"
STATS_MONTH_INCOME_KEY = "month_income"
STATS_MONTH_EXPENSES_KEY = "month_expenses"
STATS_DETAILS_KEY = "details"

DATE_SPLIT_COUNT = 3
MIN_MONTH = 1
MAX_MONTH = 12
MIN_DAY = 1
MAX_DAYS_LONG = 31
MAX_DAYS_SHORT = 30
MAX_DAYS_FEB_LEAP = 29
MAX_DAYS_FEB = 28
MONTHS_LONG = (1, 3, 5, 7, 8, 10, 12)
MONTHS_SHORT = (4, 6, 9, 11)
FEBRUARY = 2

TWO = 2
THREE = 3
FOUR = 4


DateType = tuple[int, int, int]
TransactionType = dict[str, Any]
StatsType = dict[str, Any]


def _reject_transaction(error_message: str) -> str:
    financial_transactions_storage.append({})
    return error_message


def _max_days_in_month(month: int, year: int) -> int:
    if month in MONTHS_LONG:
        return MAX_DAYS_LONG
    if month in MONTHS_SHORT:
        return MAX_DAYS_SHORT
    if month == FEBRUARY and is_leap_year(year):
        return MAX_DAYS_FEB_LEAP
    return MAX_DAYS_FEB


def _is_month_match(transaction_date: DateType, report_date: DateType) -> bool:
    month_match = transaction_date[1] == report_date[1]
    year_match = transaction_date[2] == report_date[2]
    return month_match and year_match


def date_le(date1: DateType, date2: DateType) -> bool:
    year_and_month_and_day1 = (date1[2], date1[1], date1[0])
    year_and_month_and_day2 = (date2[2], date2[1], date2[0])
    return year_and_month_and_day1 <= year_and_month_and_day2


def _collect_transaction_stats(report_date: DateType) -> StatsType:
    stats = {
        STATS_TOTAL_CAPITAL_KEY: 0,
        STATS_MONTH_INCOME_KEY: 0,
        STATS_MONTH_EXPENSES_KEY: 0,
        STATS_DETAILS_KEY: {},
    }
    for transaction in financial_transactions_storage:
        if not transaction or not date_le(transaction[TRANSACTION_DATE_KEY], report_date):
            continue
        if TRANSACTION_CATEGORY_KEY in transaction:
            _collect_cost_stats(stats, transaction, report_date)
            continue
        _collect_income_stats(stats, transaction, report_date)
    return stats


def _collect_cost_stats(stats: StatsType, transaction: TransactionType, report_date: DateType) -> None:
    amount = transaction[TRANSACTION_AMOUNT_KEY]
    stats[STATS_TOTAL_CAPITAL_KEY] -= amount
    if _is_month_match(transaction[TRANSACTION_DATE_KEY], report_date):
        stats[STATS_MONTH_EXPENSES_KEY] += amount
        category = transaction[TRANSACTION_CATEGORY_KEY]
        details = stats[STATS_DETAILS_KEY]
        details[category] = details.get(category, 0) + amount


def _collect_income_stats(stats: StatsType, transaction: TransactionType, report_date: DateType) -> None:
    amount = transaction[TRANSACTION_AMOUNT_KEY]
    stats[STATS_TOTAL_CAPITAL_KEY] += amount
    if _is_month_match(transaction[TRANSACTION_DATE_KEY], report_date):
        stats[STATS_MONTH_INCOME_KEY] += amount


def _format_detail_lines(details: dict[str, float]) -> list[str]:
    sorted_details = sorted(details.items())
    return [
        f"{count}. {category}: {amount}"
        for count, (category, amount) in enumerate(sorted_details, start=1)
    ]


def _format_stats(date: DateType, stats: StatsType) -> str:
    month_type = "loss" if stats[STATS_MONTH_INCOME_KEY] < stats[STATS_MONTH_EXPENSES_KEY] else "profit"
    month_balance = abs(stats[STATS_MONTH_INCOME_KEY] - stats[STATS_MONTH_EXPENSES_KEY])
    details_lines = _format_detail_lines(stats[STATS_DETAILS_KEY])
    lines = [
        f"Your statistics as of {date[0]}-{date[1]}-{date[2]}:",
        f"Total capital: {stats[STATS_TOTAL_CAPITAL_KEY]} rubles",
        f"This month, the {month_type} amounted to {month_balance} rubles.",
        f"Income: {stats[STATS_MONTH_INCOME_KEY]} rubles",
        f"Expenses: {stats[STATS_MONTH_EXPENSES_KEY]} rubles",
        "Details (category: amount):",
        *details_lines,
    ]
    return "\n".join(lines)


def _parse_amount(raw_value: str) -> float | None:
    try:
        return float(raw_value.replace(",", "."))
    except ValueError:
        return None


def _has_valid_date_parts(raw_date_parts: list[str]) -> bool:
    if len(raw_date_parts) != DATE_SPLIT_COUNT:
        return False
    return all(part.isdigit() for part in raw_date_parts)


def _handle_cost_command(command: list[str]) -> str:
    if len(command) == TWO and command[1] == "categories":
        return cost_categories_handler()
    if len(command) >= FOUR:
        category_name = " ".join(command[1:-2])
        amount = _parse_amount(command[-2])
        if amount is None:
            return UNKNOWN_COMMAND_MSG
        return cost_handler(category_name, amount, command[-1])
    return UNKNOWN_COMMAND_MSG


def _execute_command(command: list[str]) -> str:
    operation = command[0]
    if operation == "income" and len(command) == THREE:
        amount = _parse_amount(command[1])
        if amount is None:
            return UNKNOWN_COMMAND_MSG
        return income_handler(amount, command[2])
    if operation == "cost":
        if len(command) > THREE and command[3] in ("transport", "service"):
            command[2] = f"{command[2]} {command[3]}"
            del command[3]
        return _handle_cost_command(command)
    if operation == "stats" and len(command) == TWO:
        return stats_handler(command[1])
    return UNKNOWN_COMMAND_MSG


financial_transactions_storage: list[dict[str, Any]] = []


def is_leap_year(year: int) -> bool:
    divisible_by_four = year % 4 == 0
    divisible_by_hundred = year % 100 == 0
    divisible_by_four_hundred = year % 400 == 0
    return bool((divisible_by_four and not divisible_by_hundred) or divisible_by_four_hundred)


def extract_date(maybe_dt: str) -> tuple[int, int, int] | None:
    raw_date_parts = maybe_dt.split("-")
    if not _has_valid_date_parts(raw_date_parts):
        return None
    day, month, year = (int(part) for part in raw_date_parts)
    date: DateType = (day, month, year)
    day, month, year = date
    if not (MIN_MONTH <= month <= MAX_MONTH and day >= MIN_DAY):
        return None
    return date if day <= _max_days_in_month(month, year) else None


def income_handler(amount: float, income_date: str) -> str:
    date = extract_date(income_date)
    if amount <= 0:
        return _reject_transaction(NONPOSITIVE_VALUE_MSG)
    if date is None:
        return _reject_transaction(INCORRECT_DATE_MSG)
    financial_transactions_storage.append({
        TRANSACTION_AMOUNT_KEY: amount,
        TRANSACTION_DATE_KEY: date,
    })
    return OP_SUCCESS_MSG


def cost_handler(category_name: str, amount: float, income_date: str) -> str:
    date = extract_date(income_date)
    if amount <= 0:
        return _reject_transaction(NONPOSITIVE_VALUE_MSG)

    if date is None:
        return _reject_transaction(INCORRECT_DATE_MSG)
    contains_separator = category_name.count("::") == 1
    if not (contains_separator and all(part for part in category_name.split("::"))):
        return _reject_transaction(NOT_EXISTS_CATEGORY)

    common_category, target_category = category_name.split("::")

    if common_category not in EXPENSE_CATEGORIES or target_category not in EXPENSE_CATEGORIES[common_category]:
        return _reject_transaction(f"{NOT_EXISTS_CATEGORY}\n{cost_categories_handler()}")

    financial_transactions_storage.append({
        TRANSACTION_CATEGORY_KEY: category_name,
        TRANSACTION_AMOUNT_KEY: amount,
        TRANSACTION_DATE_KEY: date,
    })
    return OP_SUCCESS_MSG


def cost_categories_handler() -> str:
    return "\n".join(f"{common_category}::{target_category}"
                     for common_category, target_categories in EXPENSE_CATEGORIES.items()
                     for target_category in target_categories)


def stats_handler(report_date: str) -> str:
    date = extract_date(report_date)
    if date is None:
        return INCORRECT_DATE_MSG
    stats = _collect_transaction_stats(date)
    return _format_stats(date, stats)


def main() -> None:
    with open(0) as stdin:
        for line in stdin:
            print(_execute_command(line.strip().split()))


if __name__ == "__main__":
    main()
