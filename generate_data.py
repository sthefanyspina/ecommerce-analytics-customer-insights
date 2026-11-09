
import os
import numpy as np
import pandas as pd

from datetime import datetime

# ============================================================
# NOVASTORE - E-COMMERCE SYNTHETIC DATA GENERATOR
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

OUTPUT_DIR = "novastore_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

START_DATE = pd.Timestamp("2021-11-09")
END_DATE = pd.Timestamp("2026-11-08")

N_CUSTOMERS = 100_000
N_ORDERS = 500_000
N_ORDER_ITEMS = 1_000_000
N_PRODUCTS = 2_000
N_CATEGORIES = 30
N_PAYMENTS = 500_000
N_RETURNS = 30_000
N_MARKETING = 10_000

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def random_dates(start, end, size):
    """Generate random timestamps between two dates."""

    start_ts = pd.Timestamp(start).value
    end_ts = pd.Timestamp(end).value

    timestamps = rng.integers(
        start_ts,
        end_ts,
        size=size
    )

    return pd.to_datetime(timestamps)


def save_csv(dataframe, filename):
    """Save a dataframe as a CSV file."""

    filepath = os.path.join(OUTPUT_DIR, filename)

    dataframe.to_csv(
        filepath,
        index=False,
        encoding="utf-8"
    )

    print(f"Saved: {filepath} | Rows: {len(dataframe):,}")


# ============================================================
# 1. CATEGORIES
# ============================================================

print("Generating categories...")

category_names = [
    "Smartphones", "Laptops", "Desktops", "Tablets",
    "Monitors", "Keyboards", "Mice", "Headphones",
    "Speakers", "Cameras", "Smart TVs", "Printers",
    "Storage", "Memory", "Processors", "Graphics Cards",
    "Motherboards", "Power Supplies", "Networking",
    "Smartwatches", "Gaming Consoles", "Gaming Accessories",
    "Cables", "Chargers", "Smart Home", "Security",
    "Software", "Office Equipment", "Accessories", "Components"
]

categories = pd.DataFrame({
    "category_id": np.arange(1, N_CATEGORIES + 1),
    "category_name": category_names,
    "department": [
        "Electronics" if i < 10 else
        "Computing" if i < 19 else
        "Accessories" if i < 25 else
        "Technology"
        for i in range(N_CATEGORIES)
    ]
})

# ============================================================
# 2. CUSTOMERS
# ============================================================

print("Generating customers...")

first_names = [
    "Lucas", "Mariana", "Pedro", "Ana", "Gabriel",
    "Julia", "Rafael", "Camila", "Felipe", "Beatriz",
    "Bruno", "Larissa", "Matheus", "Fernanda", "Joao",
    "Isabela", "Gustavo", "Amanda", "Thiago", "Carolina"
]

last_names = [
    "Silva", "Santos", "Oliveira", "Souza", "Costa",
    "Pereira", "Rodrigues", "Almeida", "Nascimento",
    "Lima", "Araujo", "Fernandes", "Carvalho",
    "Gomes", "Martins", "Ribeiro"
]

brazilian_locations = {
    "SP": ["Sao Paulo", "Campinas", "Santos", "Ribeirao Preto"],
    "RJ": ["Rio de Janeiro", "Niteroi", "Petropolis", "Macae"],
    "MG": ["Belo Horizonte", "Uberlandia", "Contagem", "Juiz de Fora"],
    "PR": ["Curitiba", "Londrina", "Maringa", "Cascavel"],
    "RS": ["Porto Alegre", "Caxias do Sul", "Pelotas", "Santa Maria"],
    "SC": ["Florianopolis", "Joinville", "Blumenau", "Chapeco"],
    "BA": ["Salvador", "Feira de Santana", "Vitoria da Conquista"],
    "PE": ["Recife", "Olinda", "Caruaru"],
    "CE": ["Fortaleza", "Juazeiro do Norte", "Sobral"],
    "GO": ["Goiania", "Anapolis", "Rio Verde"],
    "DF": ["Brasilia"],
    "ES": ["Vitoria", "Vila Velha", "Serra"],
    "AM": ["Manaus", "Parintins"],
    "PA": ["Belem", "Santarem"],
    "MT": ["Cuiaba", "Rondonopolis"],
    "MS": ["Campo Grande", "Dourados"],
    "MA": ["Sao Luis", "Imperatriz"],
    "PB": ["Joao Pessoa", "Campina Grande"],
    "RN": ["Natal", "Mossoro"],
    "AL": ["Maceio", "Arapiraca"],
    "SE": ["Aracaju"],
    "PI": ["Teresina"],
    "TO": ["Palmas"],
    "RO": ["Porto Velho"],
    "AC": ["Rio Branco"],
    "AP": ["Macapa"],
    "RR": ["Boa Vista"]
}

states = list(brazilian_locations.keys())

# Larger states receive a higher customer probability.
state_weights = np.array([
    0.23, 0.11, 0.11, 0.06, 0.05, 0.04,
    0.07, 0.04, 0.04, 0.04, 0.02, 0.02,
    0.015, 0.015, 0.015, 0.015, 0.015,
    0.012, 0.012, 0.012, 0.01, 0.01,
    0.01, 0.01, 0.005, 0.005, 0.005
], dtype=float)

state_weights /= state_weights.sum()

customer_states = rng.choice(
    states,
    size=N_CUSTOMERS,
    p=state_weights
)

customer_cities = [
    rng.choice(brazilian_locations[state])
    for state in customer_states
]

customers = pd.DataFrame({
    "customer_id": np.arange(1, N_CUSTOMERS + 1),
    "name": [
        f"{rng.choice(first_names)} {rng.choice(last_names)}"
        for _ in range(N_CUSTOMERS)
    ],
    "gender": rng.choice(
        ["Male", "Female", "Other", None],
        size=N_CUSTOMERS,
        p=[0.48, 0.49, 0.01, 0.02]
    ),
    "birth_date": random_dates(
        "1945-01-01", "2005-12-31", N_CUSTOMERS
    ),
    "city": customer_cities,
    "state": customer_states,
    "registration_date": random_dates(
        START_DATE, END_DATE, N_CUSTOMERS
    )
})

# Introduce realistic missing data.
missing_city = rng.choice(
    N_CUSTOMERS, size=500, replace=False
)
customers.loc[missing_city, "city"] = None

missing_birth = rng.choice(
    N_CUSTOMERS, size=1000, replace=False
)
customers.loc[missing_birth, "birth_date"] = pd.NaT

# ============================================================
# 3. PRODUCTS
# ============================================================

print("Generating products...")

product_types = [
    "Premium", "Professional", "Basic",
    "Advanced", "Ultra", "Smart", "Wireless"
]

product_names = [
    "Smartphone", "Notebook", "Monitor", "Keyboard",
    "Mouse", "Headphone", "Tablet", "Camera",
    "Speaker", "SSD", "Router", "Printer",
    "Processor", "Graphics Card", "Smartwatch",
    "Webcam", "Microphone", "Power Supply",
    "Motherboard", "Gaming Controller"
]

product_category_ids = rng.integers(
    1, N_CATEGORIES + 1, N_PRODUCTS
)

costs = np.round(
    rng.lognormal(mean=5.0, sigma=0.9, size=N_PRODUCTS),
    2
)

# Selling prices are generally higher than costs.
selling_prices = np.round(
    costs * rng.uniform(1.15, 1.85, N_PRODUCTS),
    2
)

products = pd.DataFrame({
    "product_id": np.arange(1, N_PRODUCTS + 1),
    "product_name": [
        f"{rng.choice(product_types)} "
        f"{rng.choice(product_names)} {i:04d}"
        for i in range(1, N_PRODUCTS + 1)
    ],
    "category_id": product_category_ids,
    "cost": costs,
    "selling_price": selling_prices
})

# ============================================================
# 4. ORDERS
# ============================================================

print("Generating orders...")

order_ids = np.arange(1, N_ORDERS + 1)

# Customers have different purchase frequencies.
# This creates occasional buyers and high-value customers.
customer_weights = (
    1 / np.arange(1, N_CUSTOMERS + 1) ** 0.75
)

customer_weights /= customer_weights.sum()

order_customer_ids = rng.choice(
    customers["customer_id"].values,
    size=N_ORDERS,
    p=customer_weights
)

customer_registration = customers[
    "registration_date"
].values

order_customer_registration = customer_registration[
    order_customer_ids - 1
]

random_order_dates = random_dates(
    START_DATE, END_DATE, N_ORDERS
)

order_dates = pd.to_datetime(
    np.maximum(
        random_order_dates.values,
        order_customer_registration
    )
)

order_statuses = rng.choice(
    [
        "Delivered",
        "Shipped",
        "Processing",
        "Cancelled",
        "Pending"
    ],
    size=N_ORDERS,
    p=[0.88, 0.04, 0.03, 0.03, 0.02]
)

orders = pd.DataFrame({
    "order_id": order_ids,
    "customer_id": order_customer_ids,
    "order_date": order_dates,
    "status": order_statuses,
    "payment_id": order_ids
})

# ============================================================
# 5. PAYMENTS
# ============================================================

print("Generating payments...")

payment_methods = rng.choice(
    ["Credit Card", "Debit Card", "Pix", "Boleto"],
    size=N_PAYMENTS,
    p=[0.48, 0.12, 0.32, 0.08]
)

payment_statuses = np.select(
    [
        orders["status"].values == "Cancelled",
        orders["status"].values == "Pending",
        rng.random(N_PAYMENTS) < 0.04
    ],
    [
        "Refunded",
        "Pending",
        "Declined"
    ],
    default="Approved"
)

payment_dates = (
    pd.to_datetime(orders["order_date"].values)
    + pd.to_timedelta(
        rng.integers(0, 4, N_PAYMENTS),
        unit="D"
    )
)

payments = pd.DataFrame({
    "payment_id": np.arange(1, N_PAYMENTS + 1),
    "payment_method": payment_methods,
    "payment_status": payment_statuses,
    "payment_date": payment_dates
})

# ============================================================
# 6. ORDER ITEMS
# ============================================================

print("Generating order items...")

# Exactly two items per order.
item_order_ids = np.repeat(order_ids, 2)

item_product_ids = rng.integers(
    1, N_PRODUCTS + 1, N_ORDER_ITEMS
)

item_quantities = rng.choice(
    [1, 2, 3, 4],
    size=N_ORDER_ITEMS,
    p=[0.72, 0.20, 0.06, 0.02]
)

product_prices = products["selling_price"].values[
    item_product_ids - 1
]

discounts = rng.choice(
    [0, 0.05, 0.10, 0.15, 0.20, 0.30],
    size=N_ORDER_ITEMS,
    p=[0.50, 0.15, 0.15, 0.10, 0.08, 0.02]
)

order_items = pd.DataFrame({
    "order_item_id": np.arange(1, N_ORDER_ITEMS + 1),
    "order_id": item_order_ids,
    "product_id": item_product_ids,
    "quantity": item_quantities,
    "unit_price": product_prices,
    "discount": discounts
})

# ============================================================
# 7. RETURNS
# ============================================================

print("Generating returns...")

# Returns are selected from delivered orders only.
delivered_order_ids = orders.loc[
    orders["status"] == "Delivered", "order_id"
].values

eligible_items = order_items[
    order_items["order_id"].isin(delivered_order_ids)
]

selected_return_indices = rng.choice(
    eligible_items.index.values,
    size=N_RETURNS,
    replace=False
)

returned_items = eligible_items.loc[
    selected_return_indices
].copy()

returns = pd.DataFrame({
    "return_id": np.arange(1, N_RETURNS + 1),
    "order_id": returned_items["order_id"].values,
    "product_id": returned_items["product_id"].values,
    "return_date": (
        orders.set_index("order_id")
        .loc[returned_items["order_id"].values, "order_date"]
        .values
        + pd.to_timedelta(
            rng.integers(1, 31, N_RETURNS),
            unit="D"
        )
    ),
    "return_reason": rng.choice(
        [
            "Defective Product",
            "Wrong Product",
            "Changed Mind",
            "Damaged During Shipping",
            "Product Not as Described"
        ],
        size=N_RETURNS,
        p=[0.30, 0.15, 0.20, 0.20, 0.15]
    )
})

# ============================================================
# 8. MARKETING CHANNELS
# ============================================================

print("Generating marketing records...")

channels = [
    "Google Ads",
    "Facebook Ads",
    "Instagram",
    "Email Marketing",
    "Organic Search",
    "Direct",
    "Affiliate",
    "TikTok Ads",
    "YouTube Ads",
    "Referral"
]

marketing_customer_ids = rng.integers(
    1, N_CUSTOMERS + 1, N_MARKETING
)

marketing_dates = random_dates(
    START_DATE, END_DATE, N_MARKETING
)

marketing = pd.DataFrame({
    "marketing_channel_id": np.arange(1, N_MARKETING + 1),
    "customer_id": marketing_customer_ids,
    "channel": rng.choice(channels, N_MARKETING),
    "campaign": [
        f"Campaign_{x:03d}"
        for x in rng.integers(1, 51, N_MARKETING)
    ],
    "interaction_date": marketing_dates,
    "funnel_stage": rng.choice(
        [
            "Awareness",
            "Consideration",
            "Purchase",
            "Retention"
        ],
        size=N_MARKETING,
        p=[0.35, 0.30, 0.20, 0.15]
    ),
    "converted": rng.choice(
        [True, False],
        size=N_MARKETING,
        p=[0.15, 0.85]
    )
})

# ============================================================
# 9. EXPORT DATASETS
# ============================================================

print("\nExporting datasets...")

save_csv(customers, "customers.csv")
save_csv(orders, "orders.csv")
save_csv(order_items, "order_items.csv")
save_csv(products, "products.csv")
save_csv(categories, "categories.csv")
save_csv(payments, "payments.csv")
save_csv(returns, "returns.csv")
save_csv(marketing, "marketing_channels.csv")

# ============================================================
# 10. DATASET SUMMARY
# ============================================================

print("\n========================================")
print("NOVASTORE DATASET GENERATION COMPLETED")
print("========================================")

summary = {
    "customers": len(customers),
    "orders": len(orders),
    "order_items": len(order_items),
    "products": len(products),
    "categories": len(categories),
    "payments": len(payments),
    "returns": len(returns),
    "marketing_channels": len(marketing)
}

for dataset, count in summary.items():
    print(f"{dataset}: {count:,} records")

print(f"\nOutput directory: {OUTPUT_DIR}")
print("All datasets exported successfully!")
