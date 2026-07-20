temporal-sql-tool \
          --plugin postgres12 \
          --ep postgres \
          --port 5432 \
          -u temporal \
          --password temporal \
          create-database || true

        temporal-sql-tool \
          --plugin postgres12 \
          --ep postgres \
          --port 5432 \
          -u temporal \
          --password temporal \
          setup-schema -v 0.0 || true

        temporal-sql-tool \
          --plugin postgres12 \
          --ep postgres \
          --port 5432 \
          -u temporal \
          --password temporal \
          update-schema \
          -d /etc/temporal/schema/postgresql/v12/temporal/versioned

        temporal-sql-tool \
          --plugin postgres12 \
          --ep postgres \
          --port 5432 \
          --db temporal_visibility \
          -u temporal \
          --password temporal \
          create-database || true

        temporal-sql-tool \
          --plugin postgres12 \
          --ep postgres \
          --port 5432 \
          --db temporal_visibility \
          -u temporal \
          --password temporal \
          setup-schema -v 0.0 || true

        temporal-sql-tool \
          --plugin postgres12 \
          --ep postgres \
          --port 5432 \
          --db temporal_visibility \
          -u temporal \
          --password temporal \
          update-schema \
          -d /etc/temporal/schema/postgresql/v12/visibility/versioned
