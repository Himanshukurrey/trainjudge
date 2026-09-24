-- Demo shop schema. Conventions a base model won't know without examples:
--   * money is stored as integer cents (unit_price_cents), never dollars
--   * products have an is_active flag (1 = listed, 0 = discontinued)
--   * order status is one of: pending, shipped, delivered, cancelled
--   * revenue and spend exclude cancelled orders
--   * dates are ISO-8601 text (YYYY-MM-DD)

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    full_name   TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE,
    city        TEXT NOT NULL,
    signup_date TEXT NOT NULL
);

CREATE TABLE products (
    product_id       INTEGER PRIMARY KEY,
    product_name     TEXT NOT NULL UNIQUE,
    category         TEXT NOT NULL,
    unit_price_cents INTEGER NOT NULL,
    is_active        INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers (customer_id),
    order_date  TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('pending', 'shipped', 'delivered', 'cancelled'))
);

CREATE TABLE order_items (
    order_id         INTEGER NOT NULL REFERENCES orders (order_id),
    product_id       INTEGER NOT NULL REFERENCES products (product_id),
    quantity         INTEGER NOT NULL,
    unit_price_cents INTEGER NOT NULL,
    PRIMARY KEY (order_id, product_id)
);
