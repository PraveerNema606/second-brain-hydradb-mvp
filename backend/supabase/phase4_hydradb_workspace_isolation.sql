-- =============================================================================
-- Second Brain — Supabase schema (Phase 4: HydraDB workspace isolation)
-- =============================================================================
-- Apply AFTER schema.sql + phase2 + phase3. Idempotent: safe to re-run.
--
-- Goals:
--   1. Persist per-workspace HydraDB routing fields on public.workspaces
--   2. Deterministic sub-tenant ids: ws_ + first 12 hex chars of uuid (no dashes)
--   3. Backfill existing workspaces
--   4. Stamp sub-tenant + status at signup (replace handle_new_user)
--   5. Give the scheduler a hydradb_status filter ('active' by default)
--
-- The Python helper supabase_client._derived_sub_tenant_id() MUST stay in
-- sync with derive_hydradb_sub_tenant_id() below.
-- =============================================================================


-- =============================================================================
-- Columns on workspaces
-- =============================================================================
-- hydradb_sub_tenant_id / hydradb_tenant_id already exist as nullable
-- placeholders from schema.sql. Add the Phase 4 operational fields.

alter table public.workspaces
  add column if not exists hydradb_status text;

alter table public.workspaces
  add column if not exists hydradb_last_sync_at timestamptz;

-- Default + backfill status. New rows and legacy NULL rows become 'active'
-- so the scheduler's .eq("hydradb_status", "active") filter finds them.
update public.workspaces
   set hydradb_status = 'active'
 where hydradb_status is null
    or btrim(hydradb_status) = '';

alter table public.workspaces
  alter column hydradb_status set default 'active';

-- Tighten NOT NULL only after the backfill above.
do $$
begin
  alter table public.workspaces
    alter column hydradb_status set not null;
exception
  when others then
    -- Column may already be NOT NULL from a prior apply.
    null;
end$$;

-- Optional allow-list. 'paused' lets operators take a workspace out of
-- the scheduler without deleting connector state.
do $$
begin
  if not exists (
    select 1
      from pg_constraint
     where conname = 'workspaces_hydradb_status_check'
  ) then
    alter table public.workspaces
      add constraint workspaces_hydradb_status_check
      check (hydradb_status in ('active', 'paused', 'error'));
  end if;
end$$;


-- =============================================================================
-- Deterministic sub-tenant helper
-- =============================================================================
-- Format: 'ws_' || first 12 hex characters of the uuid with dashes removed.
-- Example: 11111111-aaaa-bbbb-cccc-000000000001 -> ws_11111111aaaa

create or replace function public.derive_hydradb_sub_tenant_id(wid uuid)
returns text
language sql
immutable
as $$
  select case
    when wid is null then null
    else 'ws_' || substr(replace(wid::text, '-', ''), 1, 12)
  end;
$$;

revoke all on function public.derive_hydradb_sub_tenant_id(uuid) from public;
grant execute on function public.derive_hydradb_sub_tenant_id(uuid)
  to anon, authenticated, service_role;


-- =============================================================================
-- Backfill hydradb_sub_tenant_id for existing rows
-- =============================================================================
update public.workspaces
   set hydradb_sub_tenant_id = public.derive_hydradb_sub_tenant_id(id)
 where hydradb_sub_tenant_id is null
    or btrim(hydradb_sub_tenant_id) = '';

-- Unique among non-null values so two workspaces never share a bucket.
create unique index if not exists workspaces_hydradb_sub_tenant_id_uidx
  on public.workspaces (hydradb_sub_tenant_id)
  where hydradb_sub_tenant_id is not null;


-- =============================================================================
-- Signup trigger: stamp sub-tenant + status on every new workspace
-- =============================================================================
-- Replaces the Phase 1 handle_new_user body. Behavior is identical except
-- the personal workspace insert now sets hydradb_sub_tenant_id and
-- hydradb_status so the first ingest/query never races the Python lazy
-- ensure path.

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  new_workspace_id  uuid;
  display_name      text;
  base_slug         text;
  candidate_slug    text;
  attempt           integer := 0;
begin
  display_name := coalesce(
    new.raw_user_meta_data->>'full_name',
    split_part(coalesce(new.email, ''), '@', 1),
    'user'
  );

  -- 1) profile (1:1 with auth.users)
  insert into public.profiles (id, email, full_name, avatar_url)
  values (
    new.id,
    new.email,
    display_name,
    new.raw_user_meta_data->>'avatar_url'
  )
  on conflict (id) do nothing;

  -- 2) Personal workspace. Slug derives from the email local-part, with
  --    a numeric suffix if that slug is already taken.
  base_slug := regexp_replace(
    lower(split_part(coalesce(new.email, ''), '@', 1)),
    '[^a-z0-9-]+', '-', 'g'
  );
  base_slug := trim(both '-' from coalesce(base_slug, ''));
  if base_slug = '' then
    base_slug := 'workspace';
  end if;

  candidate_slug := base_slug;
  while exists (select 1 from public.workspaces where slug = candidate_slug) loop
    attempt := attempt + 1;
    candidate_slug := base_slug || '-' || attempt::text;
  end loop;

  -- Pre-generate the workspace id so we can stamp hydradb_sub_tenant_id
  -- in the same INSERT (no NULL window, no second UPDATE).
  new_workspace_id := gen_random_uuid();

  insert into public.workspaces (
    id,
    name,
    slug,
    owner_id,
    hydradb_sub_tenant_id,
    hydradb_status
  )
  values (
    new_workspace_id,
    display_name || '''s workspace',
    candidate_slug,
    new.id,
    public.derive_hydradb_sub_tenant_id(new_workspace_id),
    'active'
  );

  -- 3) Owner membership row
  insert into public.workspace_members (workspace_id, user_id, role)
  values (new_workspace_id, new.id, 'owner');

  return new;
end$$;


-- =============================================================================
-- Smoke-check helpers (optional; safe to run in SQL editor)
-- =============================================================================
-- select public.derive_hydradb_sub_tenant_id(
--   '11111111-aaaa-bbbb-cccc-000000000001'::uuid
-- );
-- Expected: ws_11111111aaaa
--
-- select id, hydradb_sub_tenant_id, hydradb_status
--   from public.workspaces
--  where hydradb_sub_tenant_id is null;
-- Expected: 0 rows after this migration.
