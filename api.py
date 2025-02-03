from flask import Flask, request, jsonify
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)

customers = {}

class Customer:
    def __init__(self, name):
        self.name = name
        self.transaction_history = []
        self.current_balance = 0

    def add_transaction(self, transaction):
        self.transaction_history.append(transaction)
        self.current_balance += transaction.amount

    def to_dict(self):
        return {
            "name": self.name,
            "transaction_history": [t.to_dict() for t in self.transaction_history],
            "current_balance": self.current_balance
        }


class Transaction:
    def __init__(self, event_name, amount, description=""):
        self.event_name = event_name
        self.amount = amount
        self.description = description

    def to_dict(self):
        return {
            "event_name": self.event_name,
            "amount": self.amount,
            "description": self.description
        }


@app.route('/customers', methods=['GET'])
def list_customers():
    """
    List all customers.
    ---
    responses:
      200:
        description: A list of customers.
        schema:
          type: array
          items:
            type: object
            properties:
              name:
                type: string
              transaction_history:
                type: array
                items:
                  type: object
              current_balance:
                type: number
    """
    customer_list = [customer.to_dict() for customer in customers.values()]
    return jsonify(customer_list), 200


@app.route('/customers/<customer_name>/transactions', methods=['GET'])
def list_transactions(customer_name):
    """
    List transactions for a specific customer.
    ---
    parameters:
      - in: path
        name: customer_name
        type: string
        required: true
        description: The name of the customer.
    responses:
      200:
        description: A list of transactions.
        schema:
          type: array
          items:
            type: object
            properties:
              event_name:
                type: string
              amount:
                type: number
              description:
                type: string
      404:
        description: Customer not found.
    """
    customer = customers.get(customer_name)
    if not customer:
        return jsonify({"message": "Customer not found"}), 404
    return jsonify(customer.transaction_history), 200


@app.route('/customers/<customer_name>/transactions', methods=['POST'])
def add_transaction(customer_name):
    """
    Add a transaction for a specific customer.
    ---
    parameters:
      - in: path
        name: customer_name
        type: string
        required: true
        description: The name of the customer.
      - in: body
        name: transaction
        required: true
        schema:
          type: object
          properties:
            event_name:
              type: string
            amount:
              type: number
            description:
              type: string
    responses:
      201:
        description: Transaction added successfully.
      400:
        description: Invalid request body.
      404:
        description: Customer not found.
    """
    customer = customers.get(customer_name)
    if not customer:
        return jsonify({"message": "Customer not found"}), 404

    data = request.get_json()
    if not data or "event_name" not in data or "amount" not in data:
        return jsonify({"message": "Invalid request body"}), 400

    transaction = Transaction(data["event_name"], data["amount"], data.get("description", ""))
    customer.add_transaction(transaction)
    return jsonify({"message": "Transaction added successfully"}), 201


@app.route('/customers/<customer_name>/balance', methods=['GET'])
def get_balance(customer_name):
    """
    Get the current balance for a specific customer.
    ---
    parameters:
      - in: path
        name: customer_name
        type: string
        required: true
        description: The name of the customer.
    responses:
      200:
        description: The current balance.
        schema:
          type: object
          properties:
            balance:
              type: number
      404:
        description: Customer not found.
    """
    customer = customers.get(customer_name)
    if not customer:
        return jsonify({"message": "Customer not found"}), 404
    return jsonify({"balance": customer.current_balance}), 200


@app.route('/customers', methods=['POST'])
def create_customer():
    """
    Create a new customer.
    ---
    parameters:
      - in: body
        name: customer
        required: true
        schema:
          type: object
          properties:
            name:
              type: string
    responses:
      201:
        description: Customer created successfully.
      400:
        description: Customer already exists.
    """

    data = request.get_json()
    if not data or "name" not in data:
      return jsonify({"message": "Invalid request body"}), 400

    customer_name = data["name"]

    if customer_name in customers:
        return jsonify({"message": "Customer already exists"}), 400

    new_customer = Customer(customer_name)
    customers[customer_name] = new_customer
    return jsonify({"message": "Customer created successfully"}), 201

if __name__ == '__main__':
    app.run(debug=True)