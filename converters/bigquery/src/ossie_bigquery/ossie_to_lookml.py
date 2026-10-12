# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Converts an Ossie YAML semantic model to LookML, the semantic modeling
language for Looker (Google Cloud's semantic layer over BigQuery). Pure offline
conversion: no BigQuery or Looker connection required.

The converter prefers BIGQUERY-dialect expressions, then ANSI_SQL, then
OSSIE_SQL_2026, mirroring the dialect-selection contract documented in
../README.md. Expressions are parsed with sqlglot so that every column
reference can be rewritten to the LookML substitution operator (`${TABLE}.col`
inside a view, `${view.field}` across views) and so that aggregate metrics can
be typed as native LookML measures.

Mapping summary:
    Ossie document        -> one LookML document: views, then explores
    Ossie dataset         -> view (sql_table_name / derived_table, primary key)
    Ossie field           -> dimension, or dimension_group for temporal types
    Ossie metric          -> measure in the view of the dataset it aggregates
    Ossie relationship    -> explore on the many-side dataset with joins that
                             follow the relationship graph transitively
"""

import re
import warnings
from collections import Counter

import sqlglot
import yaml
from sqlglot import exp


SUPPORTED_VERSION = "0.2.0.dev0"

# Dialects the converter can emit, in order of preference.
_DIALECT_PREFERENCE = ("BIGQUERY", "ANSI_SQL", "OSSIE_SQL_2026")

# A valid LookML object name (view, field, join); also a bare SQL identifier.
_LOOKML_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Ossie `datatype` -> LookML dimension `type`.
_SCALAR_TYPES = {
    "String": "string",
    "Integer": "number",
    "Decimal": "number",
    "Float": "number",
    "Boolean": "yesno",
    # LookML has no time-of-day type; keep the value as a string.
    "Time": "string",
}

# Ossie temporal `datatype` -> LookML dimension_group `datatype`.
_TEMPORAL_TYPES = {
    "Date": "date",
    "DateTime": "datetime",
    "DateTimeTz": "timestamp",
}

_DATE_TIMEFRAMES = ["raw", "date", "week", "month", "quarter", "year"]
_TIMESTAMP_TIMEFRAMES = ["raw", "time", "date", "week", "month", "quarter", "year"]

# sqlglot aggregate node -> LookML measure type, for aggregates over a
# row-level expression (symmetric-aggregate safe in Looker).
_NATIVE_MEASURES = {
    exp.Sum: "sum",
    exp.Avg: "average",
    exp.Min: "min",
    exp.Max: "max",
}


class OssieConversionError(Exception):
    """Raised when an Ossie YAML cannot be converted to LookML."""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def convert_ossie_to_lookml(ossie_yaml_str):
    """Top-level entry point. Parses Ossie YAML, validates, converts, returns a
    LookML document as a string.

    Expects the standard Ossie document format::

        version: "0.2.0.dev0"
        name: ...
        datasets: [...]

    Args:
        ossie_yaml_str: Ossie YAML as a string.

    Returns:
        LookML document string.

    Raises:
        OssieConversionError: If the input cannot be converted.
    """
    root = yaml.safe_load(ossie_yaml_str)
    if not isinstance(root, dict):
        raise OssieConversionError("Invalid Ossie YAML: expected a mapping at the root")

    version_str = str(root.get("version", ""))
    if version_str != SUPPORTED_VERSION:
        raise OssieConversionError(
            f"Unsupported Ossie specification version '{version_str}'. "
            f"Supported: {SUPPORTED_VERSION}"
        )

    if "semantic_model" in root:
        raise OssieConversionError(
            "Legacy 'semantic_model' wrappers are not supported; "
            "place the model properties directly at the document root"
        )

    if "dialects" in root or "vendors" in root:
        raise OssieConversionError("Root dialects and vendors are not supported by the Ossie spec")

    # Document metadata is consumed here; it is not a dropped model property.
    model = {
        key: value for key, value in root.items()
        if key != "version"
    }
    return _convert_model(model)


def _convert_model(ossie):
    """Converts the root Ossie model dict to a LookML document string."""
    name = ossie.get("name")
    if not name:
        raise OssieConversionError("Missing required 'name' field in semantic model")

    datasets = ossie.get("datasets") or []
    if not datasets:
        warnings.warn(f"Model '{name}' has no datasets; output will be empty")

    plans = {}
    for dataset in datasets:
        plan = _plan_dataset(dataset)
        if plan.name in plans:
            raise OssieConversionError(f"Duplicate dataset name '{plan.name}'")
        plans[plan.name] = plan

    for metric in ossie.get("metrics") or []:
        _plan_metric(metric, plans)

    relationships = _plan_relationships(ossie.get("relationships") or [], plans)

    header = ["# Generated by the Ossie -> LookML converter", f"# Ossie model: {name}"]
    for line in _model_description(ossie).splitlines():
        header.append(f"# {line}")

    blocks = ["\n".join(header)]
    if plans:
        blocks.append("# Views (one per Ossie dataset; place in .view.lkml files)")
        blocks.extend(_render_view(plan) for plan in plans.values())
        blocks.append(
            "# Explores (place in a .model.lkml file alongside `connection` and `include`)"
        )
        blocks.extend(_render_explores(relationships, plans))

    _warn_dropped_fields(ossie, "model")

    return "\n\n".join(blocks) + "\n"


def _model_description(ossie):
    """Combines `description` and a string `ai_context.instructions` into a
    human-readable description for the model header."""
    description = ossie.get("description") or ""
    ai_context = ossie.get("ai_context")
    if isinstance(ai_context, str) and ai_context:
        description = f"{description}\n{ai_context}".strip()
    elif isinstance(ai_context, dict):
        instructions = ai_context.get("instructions")
        if isinstance(instructions, str) and instructions:
            description = f"{description}\n{instructions}".strip()
    return description


# ---------------------------------------------------------------------------
# Datasets -> views
# ---------------------------------------------------------------------------

class _Field:
    """A planned LookML field (dimension or dimension_group) for an Ossie field."""

    def __init__(self, name, lookml_name, kind, sql, column=None):
        self.name = name                # Ossie field name
        self.lookml_name = lookml_name  # dimension or dimension_group name
        self.kind = kind                # "dimension" | "group"
        self.sql = sql                  # LookML sql with ${TABLE} references
        self.column = column            # bare physical column, if the expression is one
        self.type = None                # dimension type
        self.datatype = None            # dimension_group datatype
        self.timeframes = None
        self.primary_key = False
        self.label = None
        self.description = None

    @property
    def ref(self):
        """The name other LookML references this field by: `${view.<ref>}`."""
        return f"{self.lookml_name}_raw" if self.kind == "group" else self.lookml_name


class _Measure:
    def __init__(self, name, lookml_type, sql, hidden=False, description=None):
        self.name = name
        self.type = lookml_type
        self.sql = sql
        self.hidden = hidden
        self.description = description


class _Dataset:
    """A planned LookML view for an Ossie dataset."""

    def __init__(self, dataset):
        self.name = dataset.get("name")
        if not self.name:
            raise OssieConversionError("Missing required 'name' field in dataset")
        if not _LOOKML_NAME.match(self.name):
            raise OssieConversionError(
                f"Dataset '{self.name}': name is not a valid LookML view name"
            )
        self.source = dataset.get("source")
        self.primary_key = dataset.get("primary_key") or []
        self.description = _entity_description(dataset)
        self.fields = []
        self.pk_dimension = None
        self.measures = []
        self._refs = {}

    def register(self, field):
        self.fields.append(field)
        if field.column:
            self._refs.setdefault(field.column.upper(), field)
        self._refs[field.name.upper()] = field

    def resolve(self, column):
        """Returns the planned field that exposes `column` (by field name, or
        by the bare column the field's expression selects), or None."""
        return self._refs.get(str(column).upper())


def _plan_dataset(dataset):
    plan = _Dataset(dataset)

    fields = dataset.get("fields") or []
    for field in fields:
        if not field.get("name"):
            raise OssieConversionError(
                f"Dataset '{plan.name}': missing required 'name' in field"
            )
    names = {f["name"] for f in fields}

    # A temporal field `created_at` becomes `dimension_group: created`, unless
    # the shorter name would collide with another field or another group.
    group_names = Counter()
    for field in fields:
        if _is_temporal(field):
            group_names[_strip_time_suffix(field["name"])] += 1

    for field in fields:
        name = field["name"]
        context = f"Field '{plan.name}.{name}'"
        expr = _extract_expression(field.get("expression"), f"{plan.name}.{name}")
        if expr is None:
            continue
        sql, column, _ = _qualify(expr, plan.name, {plan.name: plan}, context)

        if _is_temporal(field):
            short = _strip_time_suffix(name)
            if short != name and (short in names or group_names[short] > 1):
                short = name
            planned = _Field(name, short, "group", sql, column)
            datatype = field.get("datatype")
            planned.datatype = _TEMPORAL_TYPES.get(datatype)
            if datatype is None:
                warnings.warn(
                    f"{context}: is_time without a datatype; emitted as a time "
                    f"dimension_group (Looker assumes TIMESTAMP). Set datatype to "
                    f"Date, DateTime or DateTimeTz if the column is temporal, or "
                    f"to its scalar type if it is not"
                )
            planned.timeframes = (
                _DATE_TIMEFRAMES if planned.datatype == "date" else _TIMESTAMP_TIMEFRAMES
            )
        else:
            planned = _Field(name, name, "dimension", sql, column)
            planned.type = _dimension_type(field.get("datatype"), context)

        planned.label = field.get("label")
        planned.description = _entity_description(field)
        plan.register(planned)
        _warn_dropped_fields(field, context, skip=("label",))

    _plan_primary_key(plan)
    _warn_dropped_fields(dataset, f"dataset '{plan.name}'")
    return plan


def _is_temporal(field):
    """A field becomes a dimension_group when its datatype is temporal, or when
    it is marked is_time without a datatype. An is_time field with a scalar
    datatype (e.g. an integer year) stays a dimension."""
    datatype = field.get("datatype")
    if datatype in _TEMPORAL_TYPES:
        return True
    dimension = field.get("dimension")
    is_time = isinstance(dimension, dict) and dimension.get("is_time") is True
    return is_time and datatype is None


def _dimension_type(datatype, context):
    if datatype is None:
        return "string"
    if datatype == "Opaque":
        warnings.warn(
            f"{context}: datatype 'Opaque' has no LookML mapping; emitted as type: string"
        )
        return "string"
    lookml_type = _SCALAR_TYPES.get(datatype)
    if lookml_type is None:
        warnings.warn(
            f"{context}: unrecognized datatype '{datatype}'; emitted as type: string"
        )
        return "string"
    return lookml_type


def _plan_primary_key(plan):
    """Marks the single-column key dimension, or synthesizes a hidden
    concatenated dimension for composite keys and keys over dimension_groups
    (Looker allows primary_key on exactly one plain dimension per view)."""
    if not plan.primary_key:
        return
    if len(plan.primary_key) == 1:
        field = plan.resolve(plan.primary_key[0])
        if field is not None and field.kind == "dimension":
            field.primary_key = True
            return

    parts = [f"CAST(${{TABLE}}.{col} AS STRING)" for col in plan.primary_key]
    sql = parts[0] if len(parts) == 1 else "CONCAT(" + ", '|', ".join(parts) + ")"
    existing = {f.lookml_name for f in plan.fields}
    name = next(n for n in ("primary_key", "pk", "ossie_primary_key") if n not in existing)
    plan.pk_dimension = _Field("primary_key", name, "dimension", sql)
    plan.pk_dimension.type = "string"
    plan.pk_dimension.primary_key = True


def _render_view(plan):
    lines = [f"view: {plan.name} {{"]

    table_clause = _convert_source(plan.source, plan.name)
    if table_clause:
        lines.append(_indent(table_clause, 1))
        lines.append("")

    # LookML views have no description parameter; keep the text as a comment
    # here and emit it on the explore, which does.
    if plan.description:
        for line in plan.description.splitlines():
            lines.append(_indent(f"# {line}", 1))
        lines.append("")

    if plan.pk_dimension is not None:
        pk = plan.pk_dimension
        lines.append(_indent(
            "\n".join([
                f"dimension: {pk.lookml_name} {{",
                "  primary_key: yes",
                "  hidden: yes",
                "  type: string",
                f"  sql: {pk.sql} ;;",
                "}",
            ]), 1))
        lines.append("")

    for field in plan.fields:
        lines.append(_indent(_render_field(field), 1))
        lines.append("")

    for measure in plan.measures:
        lines.append(_indent(_render_measure(measure), 1))
        lines.append("")

    while len(lines) > 1 and lines[-1] == "":
        lines.pop()
    lines.append("}")
    return "\n".join(lines)


def _render_field(field):
    if field.kind == "group":
        lines = [f"dimension_group: {field.lookml_name} {{", "  type: time"]
        if field.datatype:
            lines.append(f"  datatype: {field.datatype}")
        lines.append(f"  timeframes: [{', '.join(field.timeframes)}]")
    else:
        lines = [f"dimension: {field.lookml_name} {{"]
        if field.primary_key:
            lines.append("  primary_key: yes")
        lines.append(f"  type: {field.type}")
    lines.append(f"  sql: {field.sql} ;;")
    if field.label:
        lines.append(f'  label: "{_escape(field.label)}"')
    if field.description:
        lines.append(f'  description: "{_escape(field.description)}"')
    lines.append("}")
    return "\n".join(lines)


def _render_measure(measure):
    lines = [f"measure: {measure.name} {{"]
    if measure.hidden:
        lines.append("  hidden: yes")
    lines.append(f"  type: {measure.type}")
    if measure.sql is not None:
        lines.append(f"  sql: {measure.sql} ;;")
    if measure.description:
        lines.append(f'  description: "{_escape(measure.description)}"')
    lines.append("}")
    return "\n".join(lines)


def _convert_source(source, dataset_name):
    """Builds the LookML table clause from an Ossie `source`.

    Returns a `sql_table_name: ... ;;` line for table references (backtick-
    quoting a 3-part BigQuery `project.dataset.table`), or a `derived_table`
    block for subquery sources. Returns None when source is empty.
    """
    if not source:
        warnings.warn(
            f"Dataset '{dataset_name}': no source provided; sql_table_name omitted"
        )
        return None

    source_stripped = str(source).strip()
    if not source_stripped:
        return None

    upper = source_stripped.upper()
    if upper.startswith(("SELECT ", "SELECT\n", "SELECT\t",
                         "WITH ", "WITH\n", "WITH\t")):
        return (
            "derived_table: {\n"
            + _indent(f"sql: {source_stripped} ;;", 1)
            + "\n}"
        )

    parts = source_stripped.split(".")
    if len(parts) == 3:
        # BigQuery fully-qualified names are conventionally backtick-quoted.
        return f"sql_table_name: `{source_stripped}` ;;"

    # Fall back to emitting the reference as-is rather than failing; Looker can
    # resolve 1- or 2-part names against the connection's default project.
    return f"sql_table_name: {source_stripped} ;;"


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

def _extract_expression(expression, name):
    """Selects the best dialect expression for BigQuery/LookML.

    Prefers BIGQUERY, then ANSI_SQL, then OSSIE_SQL_2026. Returns the
    expression string, or None if only unsupported dialects are present (the
    caller skips the entry). Raises OssieConversionError if the expression
    block is missing entirely.
    """
    if expression is None or not isinstance(expression, dict):
        raise OssieConversionError(
            f"Missing or malformed expression for field/metric '{name}'"
        )

    dialects = expression.get("dialects")
    if not dialects:
        raise OssieConversionError(f"Missing expression for field/metric '{name}'")

    available = {}
    for d in dialects:
        dialect_name = (d.get("dialect") or "").upper()
        if d.get("expression") is not None:
            available.setdefault(dialect_name, d["expression"])

    for dialect_name in _DIALECT_PREFERENCE:
        if dialect_name in available:
            return available[dialect_name]

    dialect_names = [d.get("dialect", "") for d in dialects]
    warnings.warn(
        f"Skipping field/metric '{name}': no BigQuery-compatible expression "
        f"(has dialects: {', '.join(dialect_names)}; requires one of "
        f"{', '.join(_DIALECT_PREFERENCE)})"
    )
    return None


def _parse(expr, context):
    """Parses a BigQuery expression; returns None (with a warning) when sqlglot
    cannot parse it, in which case the caller emits the text verbatim."""
    try:
        return sqlglot.parse_one(expr, dialect="bigquery")
    except sqlglot.errors.ParseError as e:
        first_line = str(e).splitlines()[0]
        warnings.warn(
            f"{context}: could not parse expression ({first_line}); emitted "
            f"verbatim, verify column references in LookML"
        )
        return None


def _qualify(expr, home, plans, context, tree=None):
    """Rewrites column references in `expr` to LookML substitutions.

    Bare `col` becomes `${TABLE}.col` (a physical column of the home view).
    `dataset.col` becomes `${dataset.<field>}`, resolved through the field
    names the dataset's view exposes (so a time column joins on its `_raw`
    timeframe).

    Returns (sql, bare_column, tree): `bare_column` is set when the expression
    is a single bare identifier; `tree` is the rewritten sqlglot tree (None
    when the expression could not be parsed).
    """
    expr = expr.strip()
    if _LOOKML_NAME.match(expr):
        return f"${{TABLE}}.{expr}", expr, exp.Var(this=f"${{TABLE}}.{expr}")

    if tree is None:
        tree = _parse(expr, context)
    if tree is None:
        return expr, None, None

    def rewrite(node):
        if not isinstance(node, exp.Column):
            return node
        column = node.name
        table = node.table
        if node.args.get("db"):
            warnings.warn(
                f"{context}: '{node.sql(dialect='bigquery')}' is a physical "
                f"reference, not an Ossie dataset.field; emitted as-is"
            )
            return exp.Var(this=node.sql(dialect="bigquery"))
        if not table:
            return exp.Var(this=f"${{TABLE}}.{column}")
        plan = plans.get(table)
        if plan is None:
            warnings.warn(
                f"{context}: '{table}.{column}' references unknown dataset "
                f"'{table}'; emitted as-is"
            )
            return exp.Var(this=f"{table}.{column}")
        field = plan.resolve(column)
        if field is not None:
            return exp.Var(this=f"${{{table}.{field.ref}}}")
        if table == home:
            warnings.warn(
                f"{context}: '{table}.{column}' matches no field in dataset "
                f"'{table}'; referencing the physical column"
            )
            return exp.Var(this=f"${{TABLE}}.{column}")
        warnings.warn(
            f"{context}: '{table}.{column}' matches no field in dataset "
            f"'{table}'; Looker needs a dimension for it"
        )
        return exp.Var(this=f"${{{table}.{column}}}")

    rewritten = tree.transform(rewrite)
    return rewritten.sql(dialect="bigquery"), None, rewritten


# ---------------------------------------------------------------------------
# Metrics -> measures
# ---------------------------------------------------------------------------

def _plan_metric(metric, plans):
    name = metric.get("name")
    if not name:
        raise OssieConversionError("Missing required 'name' in metric")
    if not _LOOKML_NAME.match(name):
        raise OssieConversionError(f"Metric '{name}': name is not a valid LookML name")
    context = f"Metric '{name}'"

    expr = _extract_expression(metric.get("expression"), name)
    if expr is None:
        return
    if not plans:
        warnings.warn(f"{context}: model has no datasets to hold the measure; skipped")
        return

    description = _entity_description(metric)
    _warn_dropped_fields(metric, context)

    tree = _parse(expr, context)
    if tree is None:
        home = next(iter(plans.values()))
        home.measures.append(_Measure(name, "number", expr.strip(), description=description))
        return

    home = _metric_home(tree, plans, context)
    sql, _, rewritten = _qualify(expr, home.name, plans, context, tree=tree)

    if rewritten.find(exp.Window) is not None:
        warnings.warn(
            f"{context}: window functions cannot be expressed as a LookML "
            f"measure; emitted as type: number with the expression verbatim"
        )
        home.measures.append(_Measure(name, "number", sql, description=description))
        return

    # A single native aggregate becomes a typed measure.
    native = _native_measure(rewritten)
    if native is not None:
        lookml_type, agg_sql = native
        home.measures.append(_Measure(name, lookml_type, agg_sql, description=description))
        return

    # Otherwise hoist each native aggregate into a hidden typed measure and
    # express the metric as arithmetic over them, so Looker's symmetric
    # aggregates still protect the aggregates across joins.
    helpers = []

    def hoist(node):
        native = _native_measure(node)
        if native is None:
            return node
        lookml_type, agg_sql = native
        helper_name = f"{name}_{lookml_type}_{len(helpers) + 1}"
        helpers.append(_Measure(helper_name, lookml_type, agg_sql, hidden=True))
        return exp.Var(this=f"${{{helper_name}}}")

    composite = rewritten.transform(hoist)
    if composite.find(exp.AggFunc) is not None:
        warnings.warn(
            f"{context}: contains an aggregate LookML cannot type natively; "
            f"emitted as type: number with the aggregate inline (not protected "
            f"by symmetric aggregates)"
        )
        home.measures.append(_Measure(name, "number", sql, description=description))
        return

    home.measures.extend(helpers)
    home.measures.append(
        _Measure(name, "number", composite.sql(dialect="bigquery"), description=description)
    )


def _native_measure(node):
    """Returns (lookml_type, sql) when `node` is an aggregate Looker can type
    natively: SUM/AVG/MIN/MAX over a row-level expression, COUNT(DISTINCT x),
    or COUNT(*). Returns None otherwise."""
    if isinstance(node, exp.Count):
        arg = node.this
        if isinstance(arg, exp.Star):
            return "count", None
        if isinstance(arg, exp.Distinct) and len(arg.expressions) == 1:
            inner = arg.expressions[0]
            if inner.find(exp.AggFunc) is None:
                return "count_distinct", inner.sql(dialect="bigquery")
        return None
    lookml_type = _NATIVE_MEASURES.get(type(node))
    if lookml_type is None or node.args.get("distinct"):
        return None
    arg = node.this
    if isinstance(arg, exp.Distinct) or arg.find(exp.AggFunc) is not None:
        return None
    return lookml_type, arg.sql(dialect="bigquery")


def _metric_home(tree, plans, context):
    """Picks the view that holds a metric's measure: the first dataset it
    qualifies a column with, else the only dataset exposing its bare columns,
    else the first dataset in the model."""
    order = list(plans)
    qualified = []
    bare = []
    for column in tree.find_all(exp.Column):
        if column.table and column.table in plans:
            if column.table not in qualified:
                qualified.append(column.table)
        elif not column.table:
            bare.append(column.name)

    if qualified:
        if len(qualified) > 1:
            warnings.warn(
                f"{context}: references datasets {', '.join(repr(q) for q in qualified)}; "
                f"emitted in view '{qualified[0]}' and only valid in explores "
                f"that join the others"
            )
        return plans[qualified[0]]

    if bare:
        owners = [
            ds for ds in order
            if all(plans[ds].resolve(col) is not None for col in bare)
        ]
        if len(owners) == 1:
            return plans[owners[0]]
        reason = "no dataset exposes" if not owners else "several datasets expose"
        warnings.warn(
            f"{context}: {reason} column(s) {', '.join(repr(b) for b in bare)}; "
            f"emitted in view '{order[0]}'. Qualify columns as dataset.field to "
            f"place the measure"
        )
        return plans[order[0]]

    warnings.warn(
        f"{context}: references no dataset columns; emitted in view '{order[0]}'"
    )
    return plans[order[0]]


# ---------------------------------------------------------------------------
# Relationships -> explores
# ---------------------------------------------------------------------------

class _Relationship:
    def __init__(self, name, from_ds, to_ds, columns):
        self.name = name
        self.from_ds = from_ds
        self.to_ds = to_ds
        self.columns = columns  # list of (from_column, to_column)


def _plan_relationships(relationships, plans):
    planned = []
    for rel in relationships:
        rel_name = rel.get("name")
        if not rel_name:
            raise OssieConversionError("Missing required 'name' field in relationship")
        context = f"Relationship '{rel_name}'"

        from_ds = rel.get("from")
        to_ds = rel.get("to")
        if not from_ds or not to_ds:
            raise OssieConversionError(f"{context}: both 'from' and 'to' are required")

        from_cols = rel.get("from_columns") or []
        to_cols = rel.get("to_columns") or []
        if not isinstance(from_cols, list) or not isinstance(to_cols, list):
            raise OssieConversionError(
                f"{context}: from_columns and to_columns must be lists of column names"
            )
        if not from_cols or not to_cols:
            raise OssieConversionError(
                f"{context}: from_columns and to_columns must each list at least one column"
            )
        if len(from_cols) != len(to_cols):
            raise OssieConversionError(
                f"{context}: from_columns and to_columns must have the same length "
                f"(got {len(from_cols)} and {len(to_cols)})"
            )

        _warn_dropped_fields(
            rel, f"relationship '{rel_name}'",
            extra=["ai_context"] if rel.get("ai_context") else None,
        )

        missing = [ds for ds in (from_ds, to_ds) if ds not in plans]
        if missing:
            for ds in missing:
                warnings.warn(f"{context}: dataset '{ds}' is not defined in the model; join skipped")
            continue

        if not plans[to_ds].primary_key:
            warnings.warn(
                f"{context}: dataset '{to_ds}' is a join target but declares no "
                f"primary_key; Looker needs one for symmetric aggregates"
            )

        planned.append(_Relationship(rel_name, from_ds, to_ds, list(zip(from_cols, to_cols))))
    return planned


def _join_ref(plan, alias, column, context):
    field = plan.resolve(column)
    if field is None:
        warnings.warn(
            f"{context}: column '{column}' matches no field in dataset "
            f"'{plan.name}'; Looker needs a dimension for it"
        )
        return f"${{{alias}.{column}}}"
    return f"${{{alias}.{field.ref}}}"


def _render_explores(relationships, plans):
    out_edges = {}
    targets = set()
    for rel in relationships:
        out_edges.setdefault(rel.from_ds, []).append(rel)
        targets.add(rel.to_ds)

    blocks = []
    for name, plan in plans.items():
        if name in out_edges:
            blocks.append(_render_explore(plan, out_edges, plans))
        elif name not in targets:
            # A view is only queryable through an explore.
            blocks.append(_render_explore(plan, {}, plans))
    return blocks


def _render_explore(base, out_edges, plans):
    """Renders one explore rooted at `base`, following many-to-one
    relationships transitively (breadth first)."""
    joins = []
    used_names = {base.name}
    # Queue entries: (dataset name, join name it was joined under, datasets on
    # the path from the base, so cycles stop).
    queue = [(base.name, base.name, (base.name,))]
    while queue:
        current, current_alias, path = queue.pop(0)
        edges = out_edges.get(current, [])
        target_counts = Counter(e.to_ds for e in edges)
        for rel in edges:
            context = f"Relationship '{rel.name}'"
            if rel.to_ds in path:
                warnings.warn(
                    f"{context}: '{rel.to_ds}' is already on the join path "
                    f"{' -> '.join(path)} in explore '{base.name}'; skipped to avoid a cycle"
                )
                continue

            if current_alias != current:
                # Under an aliased parent, children are aliased too so the two
                # paths (e.g. billing vs shipping address) stay distinct.
                join_name, aliased = f"{current_alias}_{rel.to_ds}", True
            elif target_counts[rel.to_ds] > 1:
                if not _LOOKML_NAME.match(rel.name):
                    raise OssieConversionError(
                        f"{context}: '{rel.name}' is needed as a LookML join alias "
                        f"(dataset '{rel.to_ds}' is joined more than once from "
                        f"'{current}') but is not a valid LookML name"
                    )
                join_name, aliased = rel.name, True
            else:
                join_name, aliased = rel.to_ds, False

            if join_name in used_names:
                warnings.warn(
                    f"{context}: dataset '{rel.to_ds}' is already joined as "
                    f"'{join_name}' in explore '{base.name}'; skipped"
                )
                continue
            used_names.add(join_name)

            predicates = [
                f"{_join_ref(plans[current], current_alias, fc, context)} = "
                f"{_join_ref(plans[rel.to_ds], join_name, tc, context)}"
                for fc, tc in rel.columns
            ]
            lines = [f"join: {join_name} {{"]
            if aliased:
                lines.append(f"  from: {rel.to_ds}")
            lines.append("  type: left_outer")
            lines.append("  relationship: many_to_one")
            lines.append(f"  sql_on: {' AND '.join(predicates)} ;;")
            lines.append("}")
            joins.append("\n".join(lines))
            queue.append((rel.to_ds, join_name, path + (rel.to_ds,)))

    if not joins and not base.description:
        return f"explore: {base.name} {{}}"

    lines = [f"explore: {base.name} {{"]
    if base.description:
        lines.append(_indent(f'description: "{_escape(base.description)}"', 1))
        lines.append("")
    for join in joins:
        lines.append(_indent(join, 1))
        lines.append("")
    while len(lines) > 1 and lines[-1] == "":
        lines.pop()
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _strip_time_suffix(name):
    """LookML convention: a dimension_group named `created` exposes
    `created_date`, `created_month`, etc. Strip a trailing `_date`/`_time` so
    the generated timeframes don't read as `created_date_date`."""
    for suffix in ("_date", "_time", "_datetime", "_timestamp"):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def _entity_description(entity):
    """Builds a description string from `description` plus any `ai_context`
    synonyms, mirroring the Snowflake converter's behavior."""
    description = entity.get("description") or ""
    ai_context = entity.get("ai_context")
    if isinstance(ai_context, str) and ai_context:
        description = f"{description} {ai_context}".strip()
    elif isinstance(ai_context, dict):
        synonyms = ai_context.get("synonyms")
        if isinstance(synonyms, list) and synonyms:
            joined = ", ".join(str(s) for s in synonyms)
            description = (
                f"{description} (synonyms: {joined})".strip()
                if description
                else f"synonyms: {joined}"
            )
    return description


def _escape(text):
    """Escapes a string for a double-quoted LookML value."""
    return str(text).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _indent(text, levels):
    """Indents every line of `text` by `levels` * 2 spaces."""
    pad = "  " * levels
    return "\n".join(pad + line if line else line for line in text.splitlines())


def _warn_dropped_fields(source, context, extra=None, skip=()):
    """Warns about Ossie properties that have no LookML counterpart."""
    dropped = list(extra or [])
    for key in ("custom_extensions", "version", "label"):
        if key not in skip and source.get(key):
            dropped.append(key)
    if dropped:
        warnings.warn(
            f"Dropped from {context} (no LookML counterpart): " + ", ".join(dropped)
        )
