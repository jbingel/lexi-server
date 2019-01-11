--
-- PostgreSQL database dump
--

-- Dumped from database version 10.6 (Ubuntu 10.6-0ubuntu0.18.04.1)
-- Dumped by pg_dump version 10.6 (Ubuntu 10.6-0ubuntu0.18.04.1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: blacklist; Type: TABLE; Schema: public; Owner: lexi
--

CREATE TABLE public.blacklist (
    user_id integer,
    item character varying(100)
);


ALTER TABLE public.blacklist OWNER TO lexi;

--
-- Name: models; Type: TABLE; Schema: public; Owner: lexi
--

CREATE TABLE public.models (
    user_id integer,
    year_of_birth integer,
    education character varying(10),
    model_file character varying(300),
    model_type character varying(30)
);


ALTER TABLE public.models OWNER TO lexi;

--
-- Name: TABLE models; Type: COMMENT; Schema: public; Owner: lexi
--

COMMENT ON TABLE public.models IS 'Stores information on simplification models';


--
-- Name: COLUMN models.user_id; Type: COMMENT; Schema: public; Owner: lexi
--

COMMENT ON COLUMN public.models.user_id IS 'user ID';


--
-- Name: sessions; Type: TABLE; Schema: public; Owner: lexi
--

CREATE TABLE public.sessions (
    request_id integer NOT NULL,
    user_id integer NOT NULL,
    url text,
    timestamp_start timestamp with time zone,
    timestamp_feedback timestamp with time zone,
    feedback_text character varying(1000),
    simplifications json,
    rating smallint,
    frontend_version character varying(10),
    lang_id character varying(10)
);


ALTER TABLE public.sessions OWNER TO lexi;

--
-- Name: users; Type: TABLE; Schema: public; Owner: lexi
--

CREATE TABLE public.users (
    user_id integer,
    email character varying(100),
    first_login date,
    last_login date,
    num_logins integer
);


ALTER TABLE public.users OWNER TO lexi;

--
-- Name: TABLE users; Type: COMMENT; Schema: public; Owner: lexi
--

COMMENT ON TABLE public.users IS 'Stores user and usage information';


--
-- PostgreSQL database dump complete
--

