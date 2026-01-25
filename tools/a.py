import duckdb

con = duckdb.connect()

res = con.execute("""
select approx_count_distinct(
    hash(
        struct_pack(
            x := 1::BIGINT,
            y := 'a'
        )
    )
)
""").fetchall()

print(res)
