from backend import app as backend_app

client = backend_app.app.test_client()
print('Requesting /api/market-prices/data (DATA_GOV_API_URL may be unset)')
r = client.get('/api/market-prices/data')
print('status', r.status_code)
print(r.get_json())
