## E2 API Integration

E2py provides several functions to interact with the E2 manufacturing system's JSON API. These functions allow you to authenticate, retrieve, and process data from various E2 databases such as sales, purchases, orders, jobs, and materials.

### Authentication

```python
from E2py import authHeader

# Example: Get API headers for authentication
url = "https://e2.example.com/api/"
api_key = "your_api_key"
password = "your_password"
user = "your_username"
headers = authHeader(url, api_key, password, user)
```

### Database Retrieval

- **getdB**: Generic function to retrieve any E2 database as a pandas DataFrame.

```python
from E2py import getdB

fields = "partNumber,partDescription,quantityOnHand"
inventory_df = getdB(url, headers, "inventory", fields)
```

- **longSearchQuery**: Handles large queries by batching requests.

```python
from E2py import longSearchQuery

fields = "partNumber,quantityOnHand"
part_numbers = ["PN123", "PN456", "PN789"]
results = longSearchQuery(url + "inventory", headers, fields, "partNumber", part_numbers)
```

### Sales, Purchases, and Orders

- **getSales**: Retrieve sales and invoice data.

```python
from E2py import getSales

sales_df = getSales(url, headers, "2024-01-01", root="")
```

- **getPurchases**: Retrieve purchase order data.

```python
from E2py import getPurchases

purchases_df = getPurchases(url, headers, "2024-01-01", root="")
```

- **getorderItems**: Retrieve order line items and related order info.

```python
from E2py import getorderItems

orders_df = getorderItems(url, headers, "2024-01-01", root="")
```

### Jobs and Materials

- **getJobs**: Retrieve job routings and materials.

```python
from E2py import getJobs

jobs_df, job_materials_df, estimate_materials_df = getJobs(url, headers, "2024-01-01", root="", orderItems_df=orders_df)
```

### Asynchronous Data Fetching

E2py uses `aiohttp` and `asyncio` for efficient, asynchronous data fetching from the E2 API. This allows for fast retrieval of large datasets.

---

For more details, see the docstrings in [`E2py.py`](c:/Users/zfalg/OneDrive/Desktop/E2Python/e2py_lib/E2py/E2py.py).
