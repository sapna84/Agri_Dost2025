from backend import app as backend_app

client = backend_app.app.test_client()
print('Requesting /api/market-prices/live (may attempt to fetch agmarknet)')
r = client.get('/api/market-prices/live')
print('status', r.status_code)
print(r.get_json())
