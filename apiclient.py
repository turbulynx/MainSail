import requests

class APIClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url

    def create_customer(self, name):
        url = f"{self.base_url}/customers"
        data = {"name": name}
        response = requests.post(url, json=data)
        return self._handle_response(response)

    def get_customers(self):
        url = f"{self.base_url}/customers"
        response = requests.get(url)
        return self._handle_response(response)

    def get_balance(self, customer_name):
        url = f"{self.base_url}/customers/{customer_name}/balance"
        response = requests.get(url)
        return self._handle_response(response)

    def add_transaction(self, customer_name, event_name, amount, direction, description=""):
        url = f"{self.base_url}/customers/{customer_name}/transactions"
        data = {
            "event_name": event_name,
            "amount": amount,
            "direction": direction,
            "description": description
        }
        response = requests.post(url, json=data)
        return self._handle_response(response)

    def set_hold(self, customer_name, hold_status):
        url = f"{self.base_url}/customers/{customer_name}/hold"
        data = {"hold": hold_status}
        response = requests.post(url, json=data)
        return self._handle_response(response)

    def direct_debit(self, customer_name):
        url = f"{self.base_url}/customers/{customer_name}/direct_debit"
        response = requests.post(url)
        return self._handle_response(response)

    def populate_mock_data(self):
        url = f"{self.base_url}/populate_mock_data"
        response = requests.post(url)
        return self._handle_response(response)

    def _handle_response(self, response):
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.status_code, "message": response.json().get("message", "Unknown error")}


# Main block to test all methods
if __name__ == "__main__":
    client = APIClient()

    print(client.create_customer("John Doe"))
    print(client.get_balance("John Doe"))
    print(client.add_transaction("John Doe", "Deposit", 100.0, "credit", "Initial deposit"))
    print(client.set_hold("John Doe", True))
    print(client.direct_debit("John Doe"))
    print(client.populate_mock_data())
    print(client.get_customers())
