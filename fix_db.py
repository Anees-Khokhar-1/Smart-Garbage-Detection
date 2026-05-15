"""
Safe DB inspector and fixer for the detections table used by the Flask app.

What it does:
- makes a timestamped backup of database.db
- prints current tables, schema, and sample rows
- adds missing desired columns when possible
- performs a safe migration when the table has extra or mismatched columns
"""

import os
import shutil
import sqlite3
from datetime import datetime

DB = "database.db"
DESIRED_COLS = [
    "id",
    "filename",
    "annotated_filename",
    "media_type",
    "detected_classes",
    "timestamp",
    "location",
    "incharge",
]


def backup_db(db_path: str) -> str:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"database_backup_{ts}.db"
    shutil.copyfile(db_path, backup_name)
    return backup_name


def get_tables(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [r[0] for r in cur.fetchall()]


def get_table_info(conn, table):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    return cur.fetchall()


def show_sample_rows(conn, table, limit=5):
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT * FROM {table} LIMIT {limit}")
        return cur.fetchall()
    except Exception as e:
        return f"Could not read rows from {table}: {e}"


def detection_table_sql(table_name):
    return f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id TEXT PRIMARY KEY,
        filename TEXT,
        annotated_filename TEXT,
        media_type TEXT,
        detected_classes TEXT,
        timestamp TEXT,
        location TEXT,
        incharge TEXT
    )
    """


def add_missing_columns(conn, table, missing_cols):
    cur = conn.cursor()
    for col in missing_cols:
        print(f"Adding column: {col} TEXT")
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")
    conn.commit()


def migrate_table(conn, old_table, desired_cols):
    cur = conn.cursor()
    new_table = f"{old_table}_new"
    print("Creating new table with desired schema:", new_table)
    cur.execute(detection_table_sql(new_table))

    cur.execute(f"PRAGMA table_info({old_table})")
    existing_cols = [r[1] for r in cur.fetchall()]
    print("Existing columns in old table:", existing_cols)

    select_parts = []
    for col in desired_cols:
        if col in existing_cols:
            select_parts.append(col)
        else:
            select_parts.append(f"NULL AS {col}")

    select_sql = ", ".join(select_parts)
    insert_sql = f"INSERT INTO {new_table} ({', '.join(desired_cols)}) SELECT {select_sql} FROM {old_table};"

    print("Copying data from old to new table...")
    cur.execute("BEGIN")
    try:
        cur.execute(insert_sql)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        old_backup_name = f"{old_table}_old_{ts}"
        cur.execute(f"ALTER TABLE {old_table} RENAME TO {old_backup_name}")
        cur.execute(f"ALTER TABLE {new_table} RENAME TO {old_table}")
        conn.commit()
        print(f"Migration succeeded. Old table renamed to: {old_backup_name}")
    except Exception:
        conn.rollback()
        raise


def main():
    if not os.path.exists(DB):
        print(f"Database file '{DB}' not found in current folder ({os.getcwd()}). Exiting.")
        return

    print("1) Creating backup of database...")
    backup_name = backup_db(DB)
    print("Backup created:", backup_name)

    conn = sqlite3.connect(DB)
    try:
        print("\n2) Current tables:")
        tables = get_tables(conn)
        print(tables)

        if "detections" not in tables:
            print("\nNo detections table found. Creating a new one with desired schema.")
            conn.execute(detection_table_sql("detections"))
            conn.commit()
            print("Table detections created.")
            print("\nFinal schema:")
            print(get_table_info(conn, "detections"))
            return

        print("\n3) Current detections schema:")
        info = get_table_info(conn, "detections")
        for row in info:
            print(row)
        existing_cols = [r[1] for r in info]

        print("\n4) Sample rows:")
        print(show_sample_rows(conn, "detections", limit=5))

        missing = [c for c in DESIRED_COLS if c not in existing_cols]
        extra = [c for c in existing_cols if c not in DESIRED_COLS]

        print("\nMissing desired columns:", missing)
        print("Extra columns present:", extra)

        if missing and not extra:
            print("\nOnly desired columns are missing. Adding missing columns...")
            add_missing_columns(conn, "detections", missing)
            print("Columns added.")
        elif not missing and not extra:
            print("\nSchema already matches desired schema. No action needed.")
        else:
            print("\nSchema differs. Performing safe migration...")
            migrate_table(conn, "detections", DESIRED_COLS)
            print("Migration finished.")

        print("\n5) Final detections schema:")
        for row in get_table_info(conn, "detections"):
            print(row)

        print("\n6) Sample rows after changes:")
        print(show_sample_rows(conn, "detections", limit=5))
    finally:
        conn.close()

    print("\nAll done. If something looks wrong, restore the backup:")
    print(f"  Replace {DB} with {backup_name}")


if __name__ == "__main__":
    main()
