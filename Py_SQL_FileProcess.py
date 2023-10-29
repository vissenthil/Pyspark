import mysql.connector  # pip install mysql.connector  / python -m pip install mysql.connector
import csv


def connect_db(db):
    db_connection = mysql.connector.connect(host="localhost", user="root", password="Viss@#123", database=db)
    cursor_db = db_connection.cursor()
    return cursor_db, db_connection


def execute_ddl(cursor_db):
    print(execute_ddl)
    # cursor_db.execute("CREATE TABLE IF NOT EXISTS customers (cust_id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), address VARCHAR(255))")
    cursor_db.execute('DROP TABLE IF EXISTS orders')
    cursor_db.execute('DROP TABLE IF EXISTS orders_v2')
    sql = '''CREATE TABLE IF NOT EXISTS orders (order_uniq_id INT AUTO_INCREMENT PRIMARY KEY, order_id varchar(50), cust_id varchar(50), product varchar(255), country varchar(255) , order_status varchar(50), order_amount varchar(50))'''
    sql1 = '''CREATE TABLE IF NOT EXISTS orders_v2 (order_uniq_id INT , order_id INT, cust_id INT, product varchar(100), country varchar(50) , order_status varchar(50), order_amount double)'''
    cursor_db.execute(sql)
    cursor_db.execute(sql1)
    db_connection.commit()  # save
    # db_connection.rollback() #undo
    cursor_db.close()


def csv_read(cursor_db, db_connection):
    print(csv_read)

    with open('D:\python\MLProjects\Sample.csv') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        db_connection = mysql.connector.connect(host="localhost", user="root", password="Viss@#123", database=db)
        cursor_db = db_connection.cursor()
        for row in reader:
            print(row)
            a, b, c, d, e, f = row['order_id'], row['cust_id'], row['product'], row['country'], row['order_status'], \
                               row['order_amount']
            # print(a,b,c)
            sql = "INSERT INTO orders(order_id ,cust_id, product, country, order_status, order_amount) VALUES (%s,%s,%s,%s,%s,%s)"
            cursor_db.execute(sql, [a, b, c, d, e, f])

        db_connection.commit()


def tansform_result(cursor_db, db_connection):
    #query = "insert into orders_v2 (order_uniq_id,order_id,cust_id,product,country,order_status,order_amount) SELECT order_uniq_id, cast(order_id as DECIMAL) as order_id , cast(cust_id as DECIMAL) as cust_id, cast(product as DECIMAL) as product, country, order_status, cast(order_amount as DECIMAL) as order_amount FROM orders"
    query = "insert into orders_v2 (order_uniq_id,order_id,cust_id,product,country,order_status,order_amount) SELECT order_uniq_id, order_id ,  cust_id, product, country, order_status, order_amount FROM orders"
    cursor_db.execute(query)
    db_connection.commit()
    close_connection(cursor_db, db_connection)


def select_result(cursor_db, db_connection):
    query = "SELECT * FROM orders"
    cursor_db.execute(query)
    myresult = cursor_db.fetchall()
    for x in myresult:
        print(x)
    close_connection(cursor_db, db_connection)


def close_connection(cursor_db, db_connection):
    db_connection.commit()
    cursor_db.close()
    db_connection.close()


if __name__ == "__main__":
    db = 'mydb'
    cursor_db, db_connection = connect_db(db)
    execute_ddl(cursor_db)
    close_connection(cursor_db, db_connection)
    cursor_db, db_connection = connect_db(db)
    csv_read(cursor_db, db_connection)
    # select_result(cursor_db,db_connection)
    cursor_db, db_connection = connect_db(db)
    tansform_result(cursor_db, db_connection)