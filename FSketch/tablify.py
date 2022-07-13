
class PprintTable:
    def __init__(self):
        self.margins = 10
    
    @staticmethod
    def __table__(rows, margin=10, columns=[]):
        """
        Return string representing table content, returns table as string and as a list of strings.
        It is okay for rows to have different sets of keys, table will show union of columns with
        missing values being empty spaces.
        :param rows: list of dictionaries as rows
        :param margin: left space padding to apply to each row, default is 0
        :param columns: extract listed columns in provided order, other columns will be ignored
        :return: table content as string and as list
        """

        def projection(cols, columns):
            return [(x, cols[x]) for x in columns if x in cols] if columns else cols.items()

        def row_to_string(row, columns):
            values = [
                (row[name] if name in row else "").rjust(size) for name, size in columns
            ]
            return "|%s|" % ("|".join(values))

        def header(columns):
            return "|%s|" % ("|".join([name.rjust(size) for name, size in columns]))

        def divisor(columns):
            return "+%s+" % ("+".join(["-" * size for name, size in columns]))

        data = [dict([(str(a), str(b)) for a, b in row.items()]) for row in rows]
        cols = dict([(x, len(x) + 1) for row in data for x in row.keys()]) if data else {}
        for row in data:
            for key in row.keys():
                cols[key] = max(cols[key], len(row[key]) + 1)
        proj = projection(
            cols, columns
        )  # extract certain columns to display (or all if not provided)
        table = (
            [divisor(proj), header(proj), divisor(proj)]
            + [row_to_string(row, proj) for row in data]
            + [divisor(proj)]
        )
        table = ["%s%s" % (" " * margin, tpl) for tpl in table] if margin > 0 else table
        table_text = "\n".join(table)
        return (table_text, table)

    @staticmethod
    def tablify(rows, margin=10, columns=[]):
        """
        Print table in console for list of rows.
        """
        txt, _ = __table__(rows, margin, columns)
        # txt = txt
        print(txt)

