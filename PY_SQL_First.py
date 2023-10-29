import mysql.connector

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Viss@#123"
)

print(connection)

cursor_obj = connection.cursor()

cursor_obj.execute("show databases;")

for x in cursor_obj:
    print(x)


cursor_obj.execute("use mydb;")
cursor_obj.execute("show tables;")
for tables in cursor_obj:
    print(tables)

cursor_obj.execute('Drop table if exists Customers_new')

cursor_obj.execute("show tables;")
for t in cursor_obj:
    print(t)

cursor_obj.execute('Create table if not exists Customers_new(cust_id int, cust_name varchar(200),cust_address varchar(1000));')
print('Table has been created')
query = 'insert into Customers_new values (%s,%s,%s)'
val   = (1,'senthil','mudachikkadu')
cursor_obj.execute(query,val)
connection.commit()
connection.close()


