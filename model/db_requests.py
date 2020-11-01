import psycopg2
from typing import Tuple
from model.db_tables import tables_dict


class Database:
    def __init__(self, port=5433):
        self.conn = psycopg2.connect(dbname='postgres', user='postgres', password='DOT2020', host='localhost',
                                     port=port)
        self.cursor = self.conn.cursor()

    def inserting(self, table: str, values: Tuple, columns: Tuple):
        print(len(values) == len(columns))
        columns = ", ".join(columns)
        values_string = ", ".join(["%s" for i in range(len(values))])
        self.cursor.execute(f'INSERT INTO postgres.public.{table} ({columns}) VALUES ({values_string})',
                            values)

    def selecting_with_condition(self, table: str, column: str, value):
        self.cursor.execute(f'SELECT * FROM postgres.public.{table} WHERE {column} = %s', (value, ))
        records = self.cursor.fetchall()
        return records

    def selecting(self, table: str):
        self.cursor.execute(f'SELECT * FROM postgres.public.{table}')
        records = self.cursor.fetchall()
        return records

    def selecting_by_columns(self, table: str, columns: Tuple):
        columns = ", ".join(columns)
        full_request = 'SELECT {0} FROM postgres.public.{1}'.format(columns, table)
        # print(full_request)
        self.cursor.execute(full_request)
        records = self.cursor.fetchall()
        return records

    def commiting(self):
        self.conn.commit()

    def close_conn(self):
        try:
            self.cursor.close()
            self.conn.close()
        # если соединение не было создано, то и закрывать нечего
        except AttributeError:
            pass

    def extracted_data_insert(self, text, doc_type, img_id=None):
        table = "extracteddata"
        self.inserting(table=table,
                       values=(str(text), doc_type, 4, img_id, None),
                       columns=tables_dict[table]["columns"])
        self.commiting()
        self.close_conn()


if __name__ == "__main__":
    data = Database().selecting("extracteddata")
    print(data[-1])
