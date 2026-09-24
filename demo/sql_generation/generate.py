"""Generate the SQL-generation demo dataset (deterministic).

Writes shop.sql, the seeded demo database (schema + rows) that the SQL eval
executes queries against, and data.jsonl: text-to-SQL pairs over it, with a known number of
duplicate, low-quality and malformed rows mixed in so `trainjudge audit` has
something real to find. Every clean gold query is executed against a seeded
SQLite database and must return rows.

    python demo/sql_generation/generate.py
"""

from __future__ import annotations

import json
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20260924

N_CLEAN = 1300
N_DUPLICATE = 330
N_LOW_QUALITY = 125
N_MALFORMED = 75

CITIES = [
    "Austin", "Boston", "Chicago", "Denver", "Miami", "Nashville",
    "Portland", "Phoenix", "Raleigh", "Seattle", "Tampa", "Tucson",
]
FIRST_NAMES = [
    "Ava", "Ben", "Chloe", "Dev", "Elena", "Farah", "Gabe", "Hana", "Ivan", "Jade",
    "Kofi", "Lena", "Mateo", "Nina", "Omar", "Priya", "Quinn", "Rosa", "Sam", "Tariq",
]
LAST_NAMES = ["Alvarez", "Brooks", "Chen", "Diaz", "Evans", "Fischer", "Garcia", "Haddad",
              "Ito", "Jensen"]
PRODUCTS = {
    "outdoor": ["Trailhead Daypack", "Summit Tent", "Canyon Water Filter", "Ridge Headlamp",
                "Basecamp Stove"],
    "kitchen": ["Copper Saute Pan", "Oak Cutting Board", "Pour-Over Kettle", "Chef Knife 8in",
                "Cast Iron Skillet"],
    "fitness": ["Cork Yoga Block", "Adjustable Dumbbell", "Jump Rope Pro", "Foam Roller",
                "Resistance Band Set"],
    "office": ["Walnut Desk Organizer", "Ergo Mesh Chair", "Monitor Riser", "Felt Desk Mat",
               "Brass Desk Lamp"],
    "audio": ["Studio Headphones", "Bookshelf Speakers", "Travel Earbuds", "Vinyl Turntable",
              "Soundbar Mini"],
    "garden": ["Cedar Planter Box", "Pruning Shears", "Hose Reel Cart", "Seed Starter Kit",
               "Garden Kneeler"],
    "bedding": ["Linen Duvet Cover", "Down Pillow", "Wool Throw Blanket", "Percale Sheet Set",
                "Weighted Blanket"],
    "lighting": ["Pendant Light", "Smart Bulb 4-Pack", "Floor Lamp Arc", "Solar Path Lights",
                 "LED Strip Kit"],
}
STATUSES = ["pending", "shipped", "delivered", "cancelled"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]

SCHEMA_LINE = (
    "Schema: customers(customer_id, full_name, email, city, signup_date); "
    "products(product_id, product_name, category, unit_price_cents, is_active); "
    "orders(order_id, customer_id, order_date, status); "
    "order_items(order_id, product_id, quantity, unit_price_cents)"
)

REVENUE_JOIN = (
    "FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
    "JOIN products p ON p.product_id = oi.product_id"
)
SPEND_JOIN = (
    "FROM customers c JOIN orders o ON o.customer_id = c.customer_id "
    "JOIN order_items oi ON oi.order_id = o.order_id"
)


def build_database(conn: sqlite3.Connection, rng: random.Random) -> dict:
    """Create schema.sql tables and seed them. Returns the entity lists used."""
    conn.executescript((HERE / "schema.sql").read_text())

    names = [f"{f} {last}" for f in FIRST_NAMES for last in LAST_NAMES]
    rng.shuffle(names)
    customers = names[:100]
    start = date(2024, 1, 1)
    for i, name in enumerate(customers, start=1):
        signup = start + timedelta(days=rng.randrange(0, 900))
        email = name.lower().replace(" ", ".") + "@example.com"
        conn.execute(
            "INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
            (i, name, email, rng.choice(CITIES), signup.isoformat()),
        )

    product_rows = []
    for category, items in PRODUCTS.items():
        for name in items:
            product_rows.append((name, category, rng.randrange(500, 25000, 5),
                                 0 if rng.random() < 0.1 else 1))
    for i, row in enumerate(product_rows, start=1):
        conn.execute("INSERT INTO products VALUES (?, ?, ?, ?, ?)", (i, *row))

    order_start = date(2025, 1, 1)
    for order_id in range(1, 801):
        conn.execute(
            "INSERT INTO orders VALUES (?, ?, ?, ?)",
            (
                order_id,
                rng.randrange(1, 81),  # customers 81-100 never order
                (order_start + timedelta(days=rng.randrange(0, 608))).isoformat(),
                rng.choices(STATUSES, weights=[1, 2, 6, 1])[0],
            ),
        )
        for product_id in rng.sample(range(1, len(product_rows) + 1), rng.randint(1, 4)):
            price = product_rows[product_id - 1][2]
            conn.execute(
                "INSERT INTO order_items VALUES (?, ?, ?, ?)",
                (order_id, product_id, rng.randint(1, 3), price),
            )
    conn.commit()
    return {"customers": customers, "products": [r[0] for r in product_rows]}


def candidate_pairs(entities: dict, rng: random.Random) -> list[tuple[str, str]]:
    """Every (question, gold SQL) pair the templates can produce."""
    customers = entities["customers"]
    products = entities["products"]
    dates = sorted({(date(2024, 1, 1) + timedelta(days=rng.randrange(0, 850))).isoformat()
                    for _ in range(200)})
    months = [(y, m) for y in (2025, 2026) for m in range(1, 13) if (y, m) <= (2026, 8)]
    pairs: list[tuple[str, str]] = []

    def add(questions: list[str], sql: str) -> None:
        pairs.extend((q, sql) for q in questions)

    for city in CITIES:
        add([f"How many customers are in {city}?",
             f"Count the customers located in {city}.",
             f"What's the number of customers from {city}?"],
            f"SELECT COUNT(*) FROM customers WHERE city = '{city}';")
        add([f"Which customers in {city} have never placed an order?",
             f"List {city} customers with no orders."],
            f"SELECT full_name FROM customers WHERE city = '{city}' "
            "AND customer_id NOT IN (SELECT customer_id FROM orders);")
        add([f"What is the average order value in dollars for customers in {city}?",
             f"Average order value, in dollars, for {city} customers?"],
            "SELECT AVG(order_total) / 100.0 FROM (SELECT o.order_id, "
            "SUM(oi.quantity * oi.unit_price_cents) AS order_total "
            f"{SPEND_JOIN} WHERE c.city = '{city}' AND o.status != 'cancelled' "
            "GROUP BY o.order_id);")
    for d in dates:
        add([f"List the names of customers who signed up after {d}.",
             f"Which customers joined after {d}? Give their full names.",
             f"Show full names of customers with a signup date later than {d}."],
            f"SELECT full_name FROM customers WHERE signup_date > '{d}';")
    for category in PRODUCTS:
        add([f"List active {category} products from cheapest to most expensive.",
             f"Show the {category} products still for sale, cheapest first.",
             f"Which {category} products are active? Sort by price ascending."],
            f"SELECT product_name FROM products WHERE category = '{category}' "
            "AND is_active = 1 ORDER BY unit_price_cents ASC;")
        add([f"What is the total revenue in dollars for the {category} category?",
             f"How much revenue, in dollars, has the {category} category made?"],
            f"SELECT SUM(oi.quantity * oi.unit_price_cents) / 100.0 {REVENUE_JOIN} "
            f"WHERE p.category = '{category}' AND o.status != 'cancelled';")
        for dollars in range(20, 260, 20):
            add([f"Which active {category} products cost less than ${dollars}?",
                 f"List {category} products under ${dollars} that are still for sale."],
                f"SELECT product_name FROM products WHERE category = '{category}' "
                f"AND is_active = 1 AND unit_price_cents < {dollars * 100};")
    for n in range(3, 16):
        add([f"What are the {n} most expensive products?",
             f"Show the top {n} products by price."],
            f"SELECT product_name FROM products ORDER BY unit_price_cents DESC LIMIT {n};")
        add([f"Who are the top {n} customers by total spend?",
             f"List the {n} customers who have spent the most."],
            f"SELECT c.full_name {SPEND_JOIN} WHERE o.status != 'cancelled' "
            "GROUP BY c.customer_id "
            f"ORDER BY SUM(oi.quantity * oi.unit_price_cents) DESC LIMIT {n};")
    for product in products:
        add([f"What is the price of {product} in dollars?",
             f"How much does {product} cost, in dollars?"],
            f"SELECT unit_price_cents / 100.0 FROM products WHERE product_name = '{product}';")
        add([f"How many units of {product} have been sold?",
             f"Total units sold for {product}?"],
            f"SELECT COALESCE(SUM(oi.quantity), 0) {REVENUE_JOIN} "
            f"WHERE p.product_name = '{product}' AND o.status != 'cancelled';")
    for status in STATUSES:
        add([f"How many orders are {status}?", f"Count the {status} orders."],
            f"SELECT COUNT(*) FROM orders WHERE status = '{status}';")
    for name in customers:
        add([f"How many orders has {name} placed?",
             f"Count the orders placed by {name}.",
             f"What's the order count for {name}?"],
            "SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
            f"WHERE c.full_name = '{name}';")
        add([f"What is the email address of {name}?", f"Give me {name}'s email."],
            f"SELECT email FROM customers WHERE full_name = '{name}';")
    for y, m in months:
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        add([f"How many orders were placed in {MONTHS[m - 1]} {y}?",
             f"Count orders from {MONTHS[m - 1]} {y}."],
            f"SELECT COUNT(*) FROM orders WHERE order_date >= '{y}-{m:02d}-01' "
            f"AND order_date < '{ny}-{nm:02d}-01';")
    return pairs


def _prompt(question: str) -> str:
    return f"{SCHEMA_LINE}\nQuestion: {question}"


def _row(question: str, completion) -> dict:
    return {"prompt": _prompt(question), "completion": completion}


def _duplicate_variant(row: dict, i: int) -> dict:
    question = row["prompt"].split("\nQuestion: ", 1)[1]
    variants = [
        question,
        "  " + question.replace(" ", "  ", 2) + " ",
        question.lower(),
        question.rstrip("?."),
    ]
    return _row(variants[i % 4], row["completion"])


def _low_quality(question: str, i: int) -> dict:
    kind = i % 4
    if kind == 0:
        completion = ["I'm sorry, but I can't write SQL for that request.",
                      "As an AI language model, I don't have access to your database."][i % 2]
    elif kind == 1:
        completion = ["TODO", "N/A", "...", "TBD"][i % 4]
    elif kind == 2:
        completion = question
    else:
        completion = "SELECT COUNT(*) FROM " + "FROM " * 7 + "orders;"
    return _row(question, completion)


def _malformed_line(question: str, i: int) -> str:
    kind = i % 6
    if kind == 0:
        full = json.dumps(_row(question, "SELECT 1;"))
        return full[: len(full) // 2]
    if kind == 1:
        return json.dumps({"prompt": _prompt(question)})
    if kind == 2:
        return json.dumps(_row(question, ""))
    if kind == 3:
        return json.dumps(_row(question, None))
    if kind == 4:
        return json.dumps(_row(question, ["SELECT", "*"]))
    return json.dumps([_prompt(question), "SELECT 1;"])


def main() -> None:
    rng = random.Random(SEED)
    conn = sqlite3.connect(":memory:")
    entities = build_database(conn, rng)
    db_out = HERE / "shop.sql"
    db_out.write_text("\n".join(conn.iterdump()) + "\n")

    usable = []
    seen_questions = set()
    for question, sql in candidate_pairs(entities, rng):
        if question.lower() in seen_questions:
            continue
        result = conn.execute(sql).fetchall()  # raises if the gold SQL is broken
        if not result or result == [(None,)]:
            continue
        seen_questions.add(question.lower())
        usable.append((question, sql))
    needed = N_CLEAN + N_LOW_QUALITY + N_MALFORMED
    if len(usable) < needed:
        raise SystemExit(f"only {len(usable)} usable pairs, need {needed}")
    rng.shuffle(usable)

    clean = [_row(q, sql) for q, sql in usable[:N_CLEAN]]
    rest = [q for q, _ in usable[N_CLEAN:needed]]
    lines = [json.dumps(r) for r in clean]
    lines += [json.dumps(_duplicate_variant(rng.choice(clean), i)) for i in range(N_DUPLICATE)]
    lines += [json.dumps(_low_quality(q, i)) for i, q in enumerate(rest[:N_LOW_QUALITY])]
    lines += [_malformed_line(q, i) for i, q in enumerate(rest[N_LOW_QUALITY:])]
    rng.shuffle(lines)

    out = HERE / "data.jsonl"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} rows to {out} "
          f"({N_CLEAN} clean, {N_DUPLICATE} duplicate, {N_LOW_QUALITY} low-quality, "
          f"{N_MALFORMED} malformed; {len(usable)} usable pairs available)")


if __name__ == "__main__":
    main()
