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
        self.hold = False

    def add_transaction(self, transaction):
        if self.hold:
            return False
        self.transaction_history.append(transaction)
        if transaction.direction == "credit":
            self.current_balance += transaction.amount
        elif transaction.direction == "debit":
            self.current_balance -= transaction.amount
        return True

    def set_hold(self, hold_status):
        self.hold = hold_status

    def to_dict(self):
        return {
            "name": self.name,
            "transaction_history": [t.to_dict() for t in self.transaction_history],
            "current_balance": self.current_balance,
            "hold": self.hold
        }

class Transaction:
    def __init__(self, event_name, amount, direction, description=""):
        self.event_name = event_name
        self.amount = amount
        self.direction = direction
        self.description = description

    def to_dict(self):
        return {
            "event_name": self.event_name,
            "amount": self.amount,
            "direction": self.direction,
            "description": self.description
        }

@app.route('/customers', methods=['GET'])
def list_customers():
    """Retrieve a list of all customers
    ---
    responses:
      200:
        description: A list of customers
    """
    customer_list = [customer.to_dict() for customer in customers.values()]
    return jsonify(customer_list), 200

@app.route('/customers/<customer_name>/transactions', methods=['POST'])
def add_transaction(customer_name):
    """Add a new transaction for a customer
    ---
    parameters:
      - name: customer_name
        in: path
        type: string
        required: true
        description: Name of the customer
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            event_name:
              type: string
            amount:
              type: number
            direction:
              type: string
            description:
              type: string
    responses:
      201:
        description: Transaction added successfully
      400:
        description: Invalid request body
      403:
        description: Transaction denied due to hold status
      404:
        description: Customer not found
    """
    customer = customers.get(customer_name)
    if not customer:
        return jsonify({"message": "Customer not found"}), 404

    data = request.get_json()
    if not data or "event_name" not in data or "amount" not in data or "direction" not in data:
        return jsonify({"message": "Invalid request body"}), 400

    transaction = Transaction(data["event_name"], data["amount"], data["direction"], data.get("description", ""))
    if not customer.add_transaction(transaction):
        return jsonify({"message": "Transaction denied due to hold status"}), 403
    return jsonify({"message": "Transaction added successfully"}), 201

@app.route('/customers/<customer_name>/balance', methods=['GET'])
def get_balance(customer_name):
    """Retrieve the current balance of a customer
    ---
    parameters:
      - name: customer_name
        in: path
        type: string
        required: true
        description: Name of the customer
    responses:
      200:
        description: Current balance
      404:
        description: Customer not found
    """
    customer = customers.get(customer_name)
    if not customer:
        return jsonify({"message": "Customer not found"}), 404
    return jsonify({"balance": customer.current_balance * 5}), 200

@app.route('/customers/<customer_name>/hold', methods=['POST'])
def set_hold(customer_name):
    """Set hold status for a customer
    ---
    parameters:
      - name: customer_name
        in: path
        type: string
        required: true
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            hold:
              type: boolean
    responses:
      200:
        description: Hold status updated successfully
      400:
        description: Invalid request body
      404:
        description: Customer not found
    """
    customer = customers.get(customer_name)
    if not customer:
        return jsonify({"message": "Customer not found"}), 404

    data = request.get_json()
    if "hold" not in data:
        return jsonify({"message": "Invalid request body"}), 400

    customer.set_hold(data["hold"])
    return jsonify({"message": "Hold status updated successfully"}), 200

@app.route('/customers', methods=['POST'])
def create_customer():
    """Create a new customer
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            name:
              type: string
    responses:
      201:
        description: Customer created successfully
      400:
        description: Invalid request body or customer already exists
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

@app.route('/customers/<customer_name>/direct_debit', methods=['POST'])
def direct_debit(customer_name):
    """Trigger direct debit for a customer
    ---
    parameters:
      - name: customer_name
        in: path
        type: string
        required: true
        description: Name of the customer
    responses:
      200:
        description: Direct Debit triggered successfully
      404:
        description: Customer not found
    """
    customer = customers.get(customer_name)
    if not customer:
        return jsonify({"message": "Customer not found"}), 404
    return jsonify({"message": "Direct Debit triggered successfully"}), 200

@app.route('/populate_mock_data', methods=['POST'])
def populate_mock_data():
    """Populate mock data with two customers, invoices, and random refunds
    ---
    responses:
      200:
        description: Mock data populated successfully
    """
    customer1 = Customer("Alice")
    customer2 = Customer("Bob")

    transactions = [
        Transaction("Invoice Jan", 100, "credit", "January invoice"),
        Transaction("Invoice Feb", 150, "credit", "February invoice"),
        Transaction("Invoice Mar", 200, "credit", "March invoice"),
        Transaction("Refund Feb", 50, "debit", "Partial refund February"),
        Transaction("Refund Mar", 20, "debit", "Partial refund March"),
    ]

    for t in transactions:
        customer1.add_transaction(t)

    transactions2 = [
        Transaction("Invoice Jan", 120, "credit", "January invoice"),
        Transaction("Invoice Feb", 130, "credit", "February invoice"),
        Transaction("Invoice Mar", 180, "credit", "March invoice"),
        Transaction("Refund Feb", 30, "debit", "Partial refund February"),
        Transaction("Refund Mar", 25, "debit", "Partial refund March"),
    ]

    for t in transactions2:
        customer2.add_transaction(t)

    customers["Alice"] = customer1
    customers["Bob"] = customer2

    return jsonify({"message": "Mock data populated successfully"}), 200

if __name__ == '__main__':
    app.run(debug=True)