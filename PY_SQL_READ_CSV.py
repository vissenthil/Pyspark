import csv

import mysql.connector

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Viss@#123",
    db="mydb"
)

print(connection)

cursor_obj = connection.cursor()

cursor_obj.execute('Drop table if exists Advertising')

cursor_obj.execute("CREATE TABLE IF NOT EXISTS Advertising(Srno int,TV double,radio double, newspaper double,sales double);")
csv_data = csv.reader(open('D:\\python\\MLProjects\\Advertising.csv'))
header = next(csv_data)

for row in csv_data:
    print(row)
    cursor_obj.execute('INSERT INTO Advertising(Srno,TV,radio,newspaper,sales) values(%s,%s,%s,%s,%s)',row)
connection.commit()

cursor_obj.execute('SELECT * FROM Advertising;')
Query_result = cursor_obj.fetchall()

print('Printing the selected values from the table')
for row in Query_result:
    print(row)


TABLES = {}
TABLES['employees'] = (
    "CREATE TABLE employees ("
    "  emp_no int(11) NOT NULL AUTO_INCREMENT,"
    "  birth_date date NOT NULL,"
    "  first_name varchar(14) NOT NULL,"
    "  last_name varchar(16) NOT NULL,"
    "  gender enum('M','F') NOT NULL,"
    "  hire_date date NOT NULL,"
    "  PRIMARY KEY ('emp_no')"
    ") ENGINE=InnoDB")



connection.close()

'''
TABLES['departments'] = (
    "CREATE TABLE departments` ("
    "  `dept_no` char(4) NOT NULL,"
    "  `dept_name` varchar(40) NOT NULL,"
    "  PRIMARY KEY (`dept_no`), UNIQUE KEY `dept_name` (`dept_name`)"
    ") ENGINE=InnoDB")

TABLES['salaries'] = (
    "CREATE TABLE `salaries ("
    "  emp_no int(11) NOT NULL,"
    "  salary int(11) NOT NULL,"
    "  from_date date NOT NULL,"
    "  to_date date NOT NULL,"
    "  PRIMARY KEY ('emp_no','from_date'), KEY 'emp_no' ('emp_no'),"
    "  CONSTRAINT `salaries_ibfk_1` FOREIGN KEY (`emp_no`) "
    "     REFERENCES `employees` (`emp_no`) ON DELETE CASCADE"
    ") ENGINE=InnoDB")

TABLES['dept_emp'] = (
    "CREATE TABLE `dept_emp` ("
    "  `emp_no` int(11) NOT NULL,"
    "  `dept_no` char(4) NOT NULL,"
    "  `from_date` date NOT NULL,"
    "  `to_date` date NOT NULL,"
    "  PRIMARY KEY (`emp_no`,`dept_no`), KEY `emp_no` (`emp_no`),"
    "  KEY `dept_no` (`dept_no`),"
    "  CONSTRAINT `dept_emp_ibfk_1` FOREIGN KEY (`emp_no`) "
    "     REFERENCES `employees` (`emp_no`) ON DELETE CASCADE,"
    "  CONSTRAINT `dept_emp_ibfk_2` FOREIGN KEY (`dept_no`) "
    "     REFERENCES `departments` (`dept_no`) ON DELETE CASCADE"
    ") ENGINE=InnoDB")

TABLES['dept_manager'] = (
    "  CREATE TABLE `dept_manager` ("
    "  `emp_no` int(11) NOT NULL,"
    "  `dept_no` char(4) NOT NULL,"
    "  `from_date` date NOT NULL,"
    "  `to_date` date NOT NULL,"
    "  PRIMARY KEY (`emp_no`,`dept_no`),"
    "  KEY `emp_no` (`emp_no`),"
    "  KEY `dept_no` (`dept_no`),"
    "  CONSTRAINT `dept_manager_ibfk_1` FOREIGN KEY (`emp_no`) "
    "     REFERENCES `employees` (`emp_no`) ON DELETE CASCADE,"
    "  CONSTRAINT `dept_manager_ibfk_2` FOREIGN KEY (`dept_no`) "
    "     REFERENCES `departments` (`dept_no`) ON DELETE CASCADE"
    ") ENGINE=InnoDB")

TABLES['titles'] = (
    "CREATE TABLE `titles` ("
    "  `emp_no` int(11) NOT NULL,"
    "  `title` varchar(50) NOT NULL,"
    "  `from_date` date NOT NULL,"
    "  `to_date` date DEFAULT NULL,"
    "  PRIMARY KEY (`emp_no`,`title`,`from_date`), KEY `emp_no` (`emp_no`),"
    "  CONSTRAINT `titles_ibfk_1` FOREIGN KEY (`emp_no`)"
    "     REFERENCES `employees` (`emp_no`) ON DELETE CASCADE"
    ") ENGINE=InnoDB")

'''