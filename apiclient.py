import requests
import json

from langchain_core.tools import Tool

from api import set_hold


class APIClient:
    """
    A client for interacting with the customer balance API.

    This class provides methods for creating customers, retrieving customer lists and balances,
    adding transactions, setting hold status, triggering direct debits, and populating mock data.
    """

    def __init__(self, base_url="http://localhost:5000"):
        """
        Initializes the API client with a base URL.

        Args:
            base_url (str, optional): The base URL of the API. Defaults to "http://localhost:5000".
        """
        self.base_url = base_url
    def create_customer(self, name):
        """
        Creates a new customer.

        Args:
            name (str): The name of the new customer.

        Returns:
            dict: The JSON response from the API, containing either success or error information.
        """
        url = f"{self.base_url}/customers"
        data = {"name": name}
        response = requests.post(url, json=data)
        return self._handle_response(response)


    def get_customers(self):
        """
        Retrieves a list of all customers.

        Returns:
            dict: The JSON response from the API, containing a list of customers or error information.
        """
        url = f"{self.base_url}/customers"
        response = requests.get(url)
        return self._handle_response(response)

    def get_balance(self, customer_name):
        """
        Retrieves the current balance of a customer.

        Args:
            customer_name (str): The name of the customer.

        Returns:
            dict: The JSON response from the API, containing the customer's balance or error information.
        """
        url = f"{self.base_url}/customers/{customer_name}/balance"
        response = requests.get(url)
        return self._handle_response(response)

    def add_transaction(self, customer_name, event_name, amount, direction, description=""):
        """
        Adds a new transaction for a customer.

        Args:
            customer_name (str): The name of the customer.
            event_name (str): The name of the event associated with the transaction.
            amount (float): The amount of the transaction.
            direction (str): The direction of the transaction ("credit" or "debit").
            description (str, optional): A description of the transaction. Defaults to "".

        Returns:
            dict: The JSON response from the API, containing success or error information.
        """
        url = f"{self.base_url}/customers/{customer_name}/transactions"
        data = {
            "event_name": event_name,
            "amount": amount,
            "direction": direction,
            "description": description
        }
        response = requests.post(url, json=data)
        return self._handle_response(response)

    def exit_hold(self, customer_name):
        """
        Exits the hold status for a customer.

        Args:
            customer_name (str): The name of the customer.

        Returns:
             dict: The JSON response from the API, containing success or error information.
        """
        return self.set_hold(customer_name, False)

    def set_hold(self, customer_name, value=True):
        """
        Sets the hold status for a customer.

        Args:
            customer_name (str): The name of the customer.
            value (bool): The hold status value to set (True or False).

        Returns:
             dict: The JSON response from the API, containing success or error information.
        """
        url = f"{self.base_url}/customers/{customer_name}/hold"
        data = {"hold": value}
        response = requests.post(url, json=data)
        return self._handle_response(response)

    def direct_debit(self, customer_name):
        """
        Triggers a direct debit for a customer.

        Args:
            customer_name (str): The name of the customer.

        Returns:
            dict: The JSON response from the API, containing success or error information.
        """
        url = f"{self.base_url}/customers/{customer_name}/direct_debit"
        response = requests.post(url)
        return self._handle_response(response)

    def populate_mock_data(self):
        """
        Populates the API with mock customer data.

        Returns:
            dict: The JSON response from the API, containing success or error information.
        """
        url = f"{self.base_url}/populate_mock_data"
        response = requests.post(url)
        return self._handle_response(response)

    def _handle_response(self, response):
        """
        Handles the API response, checking for errors.

        Args:
            response (requests.Response): The response object from the API.

        Returns:
            dict: The JSON response from the API.  If an error occurs, the dictionary will contain
                  an 'error' key with the status code and a 'message' key with the error message.
        """
        try:
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            error_message = response.json().get("message", str(e)) # Extract message or fall back to exception
            return {"error": response.status_code, "message": error_message}

    def call_api_method(self, request_str):
        """
        Parses a JSON string representing an API request and calls the corresponding API method.

        Args:
            request_str (str): A JSON string containing the API request.  The JSON should have
                             a "method" key specifying the API method to call, and an optional
                             "arguments" key with a dictionary of arguments for the method.  For
                             the 'add_transaction' method, the 'customer_name' should be part of the
                             'arguments' dictionary.

        Returns:
            dict or str: The API response (as a dictionary) or an error message (as a string).
        """
        try:
            request_json = json.loads(request_str)
            method_name = request_json.pop("method")
            arguments = request_json.get("arguments", {})

            if not hasattr(self, method_name):
                return f"Error: Method '{method_name}' not found in APIClient."

            method = getattr(self, method_name)

            if not isinstance(arguments, dict):
                return "Error: 'arguments' must be a dictionary."

            if method_name == "add_transaction":
                customer_name = arguments.pop('customer_name')  # Special case
                return method(customer_name, **arguments)
            else:
                return method(**arguments)

        except json.JSONDecodeError:
            return "Error: Invalid JSON input. Please provide a valid JSON string."
        except KeyError as e:
            return f"Missing required key in input: {e}"
        except TypeError as e:
            return f"Incorrect type in input: {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"


    def get_tools(self):
        return [
            Tool(
                name="List Customers",
                func=lambda _: self.get_customers(),
                description="Get a list of all customers in the system. No input required, but pass empty string if needed."
            ),
            Tool(
                name="Create Customer",
                func=self.create_customer,
                description="Create a new customer with the given name"
            ),
            Tool(
                name="Add Transaction",
                func=self._parse_transaction_input,
                description="Add a transaction for a customer_name with event_name, amount, description and direction, send the parameters as a json"
            ),
            Tool(
                name="Get Balance",
                func=self.get_balance,
                description="Get the balance for a specific customer"
            ),
            Tool(
                name="Set Hold",
                func=self.set_hold,
                description="Set a customer on hold based on customer_name"
            ),
            Tool(
                name="Exit Hold",
                func=self.set_hold,
                description="Exit hold status for a customer based on customer_name"
            ),
            Tool(
                name="Direct Debit",
                func=self.direct_debit,
                description="Trigger direct debit for a customer"
            )
        ]


    def _parse_transaction_input(self, input_str):
        import json
        try:
            input_dict = json.loads(input_str)
            customer_name = input_dict.pop('customer_name')
            return self.add_transaction(customer_name, **input_dict)
        except json.JSONDecodeError:
            return "Invalid transaction input. Please provide a valid JSON string representing a dictionary."
        except KeyError as e:
            return f"Missing required key in input: {e}"
        except TypeError as e:
            return f"Incorrect type in input: {e}"
        except Exception as e:
            return f"An error occurred: {e}"

if __name__ == "__main__":
    client = APIClient()

    while True:
        print("\nEnter your API request as a JSON string (or type 'exit'):")
        request_str = input()

        if request_str.lower() == 'exit':
            break

        result = client._call_api_method(request_str)
        print("API Response:")
        if isinstance(result, dict):
            print(json.dumps(result, indent=4))
        else:
            print(result)