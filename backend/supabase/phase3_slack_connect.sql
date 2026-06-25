-- =============================================================================
-- Second Brain — Supabase schema (Phase 3: Slack Connect)
-- =============================================================================
-- Apply AFTER schema.sql AND phase2_chat_and_saved.sql. Idempotent: safe
-- to re-run.
--
-- Creates:
--   public.slack_installations  (one row per workspace's Slack connection)
--   public.slack_channels       (channels the user can select for ingestion)
--
-- Design notes:
--
--   1. One installation per workspace. We enforce uniqueness on
--      workspace_id so Connect-Slack-twice updates the existing row
--      rather than creating duplicates. The backend reflects this by
--      using upsert on the (workspace_id) conflict target.
--
--   2. bot_token is stored in plain text for Phase 3. Encrypt-at-rest is
--      a Phase 4 concern (pgsodium / KMS). For now we mitigate via:
--        - RLS forbids SELECT on slack_installations from the frontend
--          (no policy allows authenticated reads); only the service-role
--          backend can read.
--        - The token never leaves the backend in any API response.
--
--   3. slack_channels has its own (workspace_id, slack_channel_id) unique
--      key so the backend can upsert the channel list after a Slack API
--      refresh without race-y delete-then-insert.
-- =============================================================================


-- =============================================================================
-- slack_installations
-- =============================================================================
-- Schema notes
-- ------------
-- The production deployment (Railway-hosted backend + Supabase project)
-- shipped before this file was tightened, and uses the column names
-- below. The migration is idempotent: on a fresh database it lays the
-- table down exactly; on an existing database it only adds anything
-- missing (`add column if not exists`). It does NOT drop columns, so
-- a dev database with the older `scopes` column will keep it
-- alongside the new `scope` column -- the application only writes to
-- `scope`.
create table if not exists public.slack_installations (
  id                  uuid primary key default gen_random_uuid(),
  workspace_id        uuid not null references public.workspaces(id) on delete cascade,
  slack_team_id       text not null,
  slack_team_name     text not null default '',
  slack_enterprise_id text,
  bot_user_id         text not null default '',
  bot_token           text not null,
  -- Slack OAuth v2 returns this field as `scope` (singular). Match.
  scope               text not null default '',
  installed_by        uuid references auth.users(id),
  installed_at        timestamptz not null default now(),
  revoked_at          timestamptz,
  updated_at          timestamptz not null default now(),
  unique (workspace_id)
);

-- Defensive idempotent additions: anything below covers an existing
-- database that was created BEFORE these columns existed. Each one
-- is a no-op on a fresh deploy.
alter table public.slack_installations
  add column if not exists slack_enterprise_id text;
alter table public.slack_installations
  add column if not exists scope               text not null default '';
alter table public.slack_installations
  add column if not exists installed_by        uuid references auth.users(id);
alter table public.slack_installations
  add column if not exists installed_at        timestamptz not null default now();
alter table public.slack_installations
  add column if not exists revoked_at          timestamptz;

-- Unique by (workspace_id, slack_team_id) is also valid here -- the
-- application enforces single-installation-per-workspace via the
-- bare `unique (workspace_id)` constraint above, but adding this is
-- harmless and matches the production schema.
do $$
begin
  begin
    alter table public.slack_installations
      add constraint slack_installations_workspace_team_key
      unique (workspace_id, slack_team_id);
  exception when others then
    -- Already exists (under this name or a different name); ignore.
    null;
  end;
end$$;

create index if not exists slack_installations_workspace_idx
  on public.slack_installations (workspace_id);

drop trigger if exists slack_installations_set_updated_at
  on public.slack_installations;
create trigger slack_installations_set_updated_at
  before update on public.slack_installations
  for each row execute function public.set_updated_at();


-- =============================================================================
-- slack_channels
-- =============================================================================
-- Mirrors the production schema. Dev databases that were created
-- before these columns existed pick them up via the idempotent
-- `add column if not exists` block below. `installation_id` is set
-- by the backend when it has the row in hand; the column is
-- nullable in this migration so legacy callers continue to work,
-- but the production schema treats it as required and the backend
-- always passes it.
create table if not exists public.slack_channels (
  id                uuid primary key default gen_random_uuid(),
  workspace_id      uuid not null references public.workspaces(id) on delete cascade,
  installation_id   uuid references public.slack_installations(id) on delete cascade,
  slack_channel_id  text not null,
  name              text not null default '',
  is_private        boolean not null default false,
  is_archived       boolean not null default false,
  member_count      integer not null default 0,
  topic             text not null default '',
  purpose           text not null default '',
  is_selected       boolean not null default false,
  last_seen_at      timestamptz,
  updated_at        timestamptz not null default now(),
  unique (workspace_id, slack_channel_id)
);

-- Defensive idempotent additions: anything below covers an existing
-- database that was created before these columns existed. Each one
-- is a no-op on a fresh deploy.
alter table public.slack_channels
  add column if not exists installation_id uuid references public.slack_installations(id) on delete cascade;
alter table public.slack_channels
  add column if not exists is_private      boolean not null default false;
alter table public.slack_channels
  add column if not exists member_count    integer not null default 0;
alter table public.slack_channels
  add column if not exists topic           text    not null default '';
alter table public.slack_channels
  add column if not exists purpose         text    not null default '';
alter table public.slack_channels
  add column if not exists last_seen_at    timestamptz;
alter table public.slack_channels
  add column if not exists bot_removed          boolean not null default false;
alter table public.slack_channels
  add column if not exists include_bot_messages boolean not null default false;

-- The production schema also has UNIQUE(installation_id, slack_channel_id).
-- Add it defensively for parity.
do $$
begin
  begin
    alter table public.slack_channels
      add constraint slack_channels_installation_channel_key
      unique (installation_id, slack_channel_id);
  exception when others then
    null;
  end;
end$$;

create index if not exists slack_channels_workspace_idx
  on public.slack_channels (workspace_id);
create index if not exists slack_channels_installation_idx
  on public.slack_channels (installation_id);

drop trigger if exists slack_channels_set_updated_at
  on public.slack_channels;
create trigger slack_channels_set_updated_at
  before update on public.slack_channels
  for each row execute function public.set_updated_at();


-- =============================================================================
-- Row-level security
-- =============================================================================
alter table public.slack_installations enable row level security;
alter table public.slack_channels      enable row level security;


-- slack_installations -----------------------------------------------------
-- Read: NO policy grants authenticated SELECT access. The frontend
-- never reads this table — bot_token must stay server-side. The
-- service-role key (used by the backend) bypasses RLS.
-- Write: same. All writes go through the backend.
--
-- We deliberately do NOT create any policies here. With RLS enabled
-- and no policies, all authenticated-role queries fail closed — exactly
-- what we want.

-- slack_channels ----------------------------------------------------------
-- Read: any workspace member can list channels for their workspace
-- (no secrets in this table; this is what powers the channel-picker UI
-- if a future client ever talks to Supabase directly).
drop policy if exists slack_channels_member_select on public.slack_channels;
create policy slack_channels_member_select on public.slack_channels
  for select using (public.user_is_workspace_member(workspace_id));

-- Write: no policy — only the backend writes (after Slack OAuth
-- completion and after user toggles selection). The service-role key
-- bypasses RLS.