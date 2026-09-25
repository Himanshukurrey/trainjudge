"""E-commerce and retail."""

from trainjudge.domains import DomainPack

PACK = DomainPack(
    name="ecommerce",
    label="E-commerce/retail",
    description="Online and physical retail, marketplaces and product catalogs",
    terms=(
        "product",
        "products",
        "catalog",
        "catalogue",
        "sku",
        "skus",
        "inventory",
        "cart",
        "checkout",
        "shopping",
        "shop",
        "store",
        "orders",
        "shipping",
        "returns",
        "refunds",
        "merchant",
        "merchants",
        "marketplace",
        "product listing",
        "product listings",
        "product description",
        "product descriptions",
        "pricing",
        "promotions",
        "coupon",
        "coupons",
        "e-commerce",
        "ecommerce",
        "retail",
        "retailer",
    ),
    changing_fact_terms=(
        "prices",
        "pricing",
        "stock",
        "inventory",
        "availability",
        "promotions",
        "discounts",
        "coupons",
        "shipping rates",
        "delivery times",
        "return policy",
        "catalog",
        "product details",
        "specifications",
    ),
    changing_facts="price changes, stock levels, promotions and catalog updates",
    changing_fact_note=(
        "change daily. Look them up from the live catalog or a retrieval index when answering; a "
        "model trained on last month's prices will quote them confidently."
    ),
    retrieval_hint=" (or query the live catalog)",
    high_stakes_terms=(
        "fraud decision",
        "fraud decisions",
        "account suspension",
        "seller suspension",
        "suspend accounts",
        "ban sellers",
        "refund denial",
        "deny refunds",
    ),
    high_stakes_note=(
        "Keep a human review step before acting on customers or sellers, give them a way to "
        "appeal, and check that outcomes aren't skewed against particular groups or regions."
    ),
)
