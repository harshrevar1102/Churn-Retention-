-- Cohort retention: for each acquisition month (first purchase), how many
-- customers were active N months later?  Table `sales` = cleaned sales lines.
WITH first_purchase AS (
    SELECT CustomerID, strftime('%Y-%m', MIN(InvoiceDate)) AS cohort_month
    FROM sales
    GROUP BY CustomerID
),
monthly_activity AS (
    SELECT DISTINCT CustomerID, strftime('%Y-%m', InvoiceDate) AS active_month
    FROM sales
)
SELECT
    f.cohort_month,
    (CAST(substr(a.active_month, 1, 4) AS INTEGER) - CAST(substr(f.cohort_month, 1, 4) AS INTEGER)) * 12
      + CAST(substr(a.active_month, 6, 2) AS INTEGER) - CAST(substr(f.cohort_month, 6, 2) AS INTEGER)
      AS month_index,
    COUNT(DISTINCT a.CustomerID) AS active_customers
FROM first_purchase f
JOIN monthly_activity a ON a.CustomerID = f.CustomerID
GROUP BY f.cohort_month, month_index
ORDER BY f.cohort_month, month_index;
