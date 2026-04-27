#ifndef __TOOLS_COMMON_INCLUDE_GETOPT_H__
/**
 * DISCLAIMER
 * This file is part of the mingw-w64 runtime package.
 *
 * The mingw-w64 runtime package and its code is distributed in the hope that it
 * will be useful but WITHOUT ANY WARRANTY.  ALL WARRANTIES, EXPRESSED OR
 * IMPLIED ARE HEREBY DISCLAIMED.  This includes but is not limited to
 * warranties of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
/*
 * Copyright (c) 2002 Todd C. Miller <Todd.Miller@courtesan.com>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 * Sponsored in part by the Defense Advanced Research Projects
 * Agency (DARPA) and Air Force Research Laboratory, Air Force
 * Materiel Command, USAF, under agreement number F39502-99-1-0512.
 */
/*-
 * Copyright (c) 2000 The NetBSD Foundation, Inc.
 * All rights reserved.
 *
 * This code is derived from software contributed to The NetBSD Foundation
 * by Dieter Baron and Thomas Klausner.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE NETBSD FOUNDATION, INC. AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma warning(disable : 4996);

#define __TOOLS_COMMON_INCLUDE_GETOPT_H__

#include <crtdefs.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

#define REPLACE_GETOPT

#ifdef REPLACE_GETOPT
int opterr = 1;
int optind = 1;
int optopt = '?';
#undef optreset
#define optreset __mingw_optreset
int optreset;
char *optarg;
#endif

#define PRINT_ERROR ((opterr) && (*options != ':'))

// permute nonoptions to the end of the argv array.
#define FLAG_PERMUTE 0x01
// treat non-options as args to option "-1" (GNU extension)
#define FLAG_ALLARGS 0x02
// operate as getopt_long_only, which means that long-named options start with a
// single '-' and are disambiguated from short-named options by their longer
// length.
#define FLAG_LONGONLY 0x04

/* return values */
#define BADCH (int)'?'
#define BADARG ((*options == ':') ? (int)':' : (int)'?')
#define INORDER (int)1

#ifndef __CYGWIN__
#define __progname __argv[0]
#else
extern char __declspec(dllimport) * __progname;
#endif

#ifdef __CYGWIN__
static char EMSG[] = "";
#else
#define EMSG ""
#endif

static int getopt_internal(int, char *const *, const char *,
                           const struct option *, int *, int);
static int parse_long_options(char *const *, const char *,
                              const struct option *, int *, int);
static int gcd(int, int);
static void permute_args(int, int, int, char *const *);

static char *option_cursor = EMSG;

// XXX: set optreset to 1 rather than these two
static int first_nonopt_index = -1;
static int first_opt_after_nonopt_index = -1;

// Error message for option that requires an argument
static const char recargchar[] = "option requires an argument -- %c";
static const char recargstring[] = "option requires an argument -- %s";
static const char ambig[] = "ambiguous option -- %.*s";
static const char noarg[] = "option doesn't take an argument -- %.*s";
static const char illoptchar[] = "unknown option -- %c";
static const char illoptstring[] = "unknown option -- %s";

static void _vwarnx(const char *fmt, va_list ap) {
  (void)fprintf(stderr, "%s: ", __progname);
  if (fmt != NULL) (void)vfprintf(stderr, fmt, ap);
  (void)fprintf(stderr, "\n");
}

static void warnx(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  _vwarnx(fmt, ap);
  va_end(ap);
}

/*
 * Compute the greatest common divisor of a and b.
 */
static int gcd(int a, int b) {
  int c;

  c = a % b;
  while (c != 0) {
    a = b;
    b = c;
    c = a % b;
  }

  return (b);
}

/*
 * Exchange the block from first_nonopt_index to first_opt_after_nonopt_index
 * with the block from first_opt_after_nonopt_index to opt_end
 * (keeping the same order of arguments
 * in each block).
 */
static void permute_args(int panonopt_start, int panonopt_end, int opt_end,
                         char *const *nargv) {
  int cstart, cyclelen, i, j, ncycle, nnonopts, nopts, pos;
  char *swap;

  nnonopts = panonopt_end - panonopt_start;
  nopts = opt_end - panonopt_end;
  ncycle = gcd(nnonopts, nopts);
  cyclelen = (opt_end - panonopt_start) / ncycle;

  for (i = 0; i < ncycle; i++) {
    cstart = panonopt_end + i;
    pos = cstart;
    for (j = 0; j < cyclelen; j++) {
      if (pos >= panonopt_end)
        pos -= nnonopts;
      else
        pos += nopts;
      swap = nargv[pos];
      /* LINTED const cast */
      ((char **)nargv)[pos] = nargv[cstart];
      /* LINTED const cast */
      ((char **)nargv)[cstart] = swap;
    }
  }
}

#ifdef REPLACE_GETOPT
int getopt(int nargc, char *const *nargv, const char *options) {
  return (getopt_internal(nargc, nargv, options, NULL, NULL, 0));
}
#endif

#ifdef _BSD_SOURCE
#define optreset __mingw_optreset
extern int optreset;
#endif
#ifdef __cplusplus
}
#endif

#endif

#if !defined(__UNISTD_H_SOURCED__) && !defined(__GETOPT_LONG_H__)
#define __GETOPT_LONG_H__

#ifdef __cplusplus
extern "C" {
#endif

struct option /* specification for a long form option...	*/
{
  const char *name; /* option name, without leading hyphens */
  int has_arg;      /* does it take an argument?		*/
  int *flag;        /* where to save its status, or NULL	*/
  int val;          /* its associated status value		*/
};

enum                 /* permitted values for its `has_arg' field...	*/
{ no_argument = 0,   /* option never takes an argument	*/
  required_argument, /* option always requires an argument	*/
  optional_argument  /* option may take an argument		*/
};

static int parse_long_options(char *const *nargv, const char *options,
                              const struct option *long_options, int *idx,
                              int short_too) {
  char *current_arg, *equal_sign;
  size_t current_arg_len;
  int i, is_ambiguous, matched_option;

#define IDENTICAL_INTERPRETATION(_x, _y)                       \
  (long_options[(_x)].has_arg == long_options[(_y)].has_arg && \
   long_options[(_x)].flag == long_options[(_y)].flag &&       \
   long_options[(_x)].val == long_options[(_y)].val)

  current_arg = option_cursor;
  matched_option = -1;
  is_ambiguous = 0;

  optind++;

  if ((equal_sign = strchr(current_arg, '=')) != NULL) {
    /* argument found (--option=arg) */
    current_arg_len = equal_sign - current_arg;
    equal_sign++;
  } else
    current_arg_len = strlen(current_arg);

  for (i = 0; long_options[i].name; i++) {
    /* find matching long option */
    if (strncmp(current_arg, long_options[i].name, current_arg_len)) continue;

    if (strlen(long_options[i].name) == current_arg_len) {
      /* exact match */
      matched_option = i;
      is_ambiguous = 0;
      break;
    }
    /*
     * If this is a known short option, don't allow
     * a partial match of a single character.
     */
    if (short_too && current_arg_len == 1) continue;

    if (matched_option == -1) /* partial match */
      matched_option = i;
    else if (!IDENTICAL_INTERPRETATION(i, matched_option))
      is_ambiguous = 1;
  }
  if (is_ambiguous) {
    /* ambiguous abbreviation */
    if (PRINT_ERROR) warnx(ambig, (int)current_arg_len, current_arg);
    optopt = 0;
    return (BADCH);
  }
  if (matched_option != -1) { /* option found */
    if (long_options[matched_option].has_arg == no_argument && equal_sign) {
      if (PRINT_ERROR) warnx(noarg, (int)current_arg_len, current_arg);
      /*
       * XXX: GNU sets optopt to val regardless of flag
       */
      if (long_options[matched_option].flag == NULL)
        optopt = long_options[matched_option].val;
      else
        optopt = 0;
      return (BADARG);
    }
    if (long_options[matched_option].has_arg == required_argument ||
        long_options[matched_option].has_arg == optional_argument) {
      if (equal_sign)
        optarg = equal_sign;
      else if (long_options[matched_option].has_arg == required_argument) {
        /*
         * optional argument doesn't use next nargv
         */
        optarg = nargv[optind++];
      }
    }
    if ((long_options[matched_option].has_arg == required_argument) &&
        (optarg == NULL)) {
      /*
       * Missing argument; leading ':' indicates no error
       * should be generated.
       */
      if (PRINT_ERROR) warnx(recargstring, current_arg);
      /*
       * XXX: GNU sets optopt to val regardless of flag
       */
      if (long_options[matched_option].flag == NULL)
        optopt = long_options[matched_option].val;
      else
        optopt = 0;
      --optind;
      return (BADARG);
    }
  } else { /* unknown option */
    if (short_too) {
      --optind;
      return (-1);
    }
    if (PRINT_ERROR) warnx(illoptstring, current_arg);
    optopt = 0;
    return (BADCH);
  }
  if (idx) *idx = matched_option;
  if (long_options[matched_option].flag) {
    *long_options[matched_option].flag = long_options[matched_option].val;
    return (0);
  } else
    return (long_options[matched_option].val);
#undef IDENTICAL_INTERPRETATION
}

/*
 * getopt_internal --
 *	Parse argc/argv argument vector.  Called by user level routines.
 */
static int getopt_internal(int nargc, char *const *nargv, const char *options,
                           const struct option *long_options, int *idx,
                           int flags) {
  char *option_spec; /* option letter list index */
  int current_optchar, allow_short_match;
  static int posixly_correct = -1;

  if (options == NULL) return (-1);

  if (optind == 0) optind = optreset = 1;

  if (posixly_correct == -1 || optreset != 0)
    posixly_correct = (getenv("POSIXLY_CORRECT") != NULL);
  if (*options == '-')
    flags |= FLAG_ALLARGS;
  else if (posixly_correct || *options == '+')
    flags &= ~FLAG_PERMUTE;
  if (*options == '+' || *options == '-') options++;

  optarg = NULL;
  if (optreset) first_nonopt_index = first_opt_after_nonopt_index = -1;
start:
  if (optreset || !*option_cursor) {
    optreset = 0;
    if (optind >= nargc) {
      option_cursor = EMSG;
      if (first_opt_after_nonopt_index != -1) {
        permute_args(first_nonopt_index, first_opt_after_nonopt_index, optind,
                     nargv);
        optind -= first_opt_after_nonopt_index - first_nonopt_index;
      } else if (first_nonopt_index != -1) {
        optind = first_nonopt_index;
      }
      first_nonopt_index = first_opt_after_nonopt_index = -1;
      return (-1);
    }
    if (*(option_cursor = nargv[optind]) != '-' ||
        (option_cursor[1] == '\0' && strchr(options, '-') == NULL)) {
      option_cursor = EMSG; /* found non-option */
      if (flags & FLAG_ALLARGS) {
        optarg = nargv[optind++];
        return (INORDER);
      }
      if (!(flags & FLAG_PERMUTE)) {
        return (-1);
      }
      /* do permutation */
      if (first_nonopt_index == -1)
        first_nonopt_index = optind;
      else if (first_opt_after_nonopt_index != -1) {
        permute_args(first_nonopt_index, first_opt_after_nonopt_index, optind,
                     nargv);
        first_nonopt_index =
            optind - (first_opt_after_nonopt_index - first_nonopt_index);
        first_opt_after_nonopt_index = -1;
      }
      optind++;
      goto start;
    }
    if (first_nonopt_index != -1 && first_opt_after_nonopt_index == -1)
      first_opt_after_nonopt_index = optind;

    if (option_cursor[1] != '\0' && *++option_cursor == '-' &&
        option_cursor[1] == '\0') {
      optind++;
      option_cursor = EMSG;

      if (first_opt_after_nonopt_index != -1) {
        permute_args(first_nonopt_index, first_opt_after_nonopt_index, optind,
                     nargv);
        optind -= first_opt_after_nonopt_index - first_nonopt_index;
      }
      first_nonopt_index = first_opt_after_nonopt_index = -1;
      return (-1);
    }
  }

  if (long_options != NULL && option_cursor != nargv[optind] &&
      (*option_cursor == '-' || (flags & FLAG_LONGONLY))) {
    allow_short_match = 0;
    if (*option_cursor == '-')
      option_cursor++;
    else if (*option_cursor != ':' && strchr(options, *option_cursor) != NULL)
      allow_short_match = 1;

    current_optchar = parse_long_options(nargv, options, long_options, idx,
                                         allow_short_match);
    if (current_optchar != -1) {
      option_cursor = EMSG;
      return (current_optchar);
    }
  }

  if ((current_optchar = (int)*option_cursor++) == (int)':' ||
      (current_optchar == (int)'-' && *option_cursor != '\0') ||
      (option_spec = (char *)strchr(options, current_optchar)) == NULL) {
    if (current_optchar == (int)'-' && *option_cursor == '\0') return (-1);
    if (!*option_cursor) ++optind;
    if (PRINT_ERROR) warnx(illoptchar, current_optchar);
    optopt = current_optchar;
    return (BADCH);
  }
  if (long_options != NULL && current_optchar == 'W' && option_spec[1] == ';') {
    if (*option_cursor)
      /* NOTHING */;
    else if (++optind >= nargc) {
      option_cursor = EMSG;
      if (PRINT_ERROR) warnx(recargchar, current_optchar);
      optopt = current_optchar;
      return (BADARG);
    } else
      option_cursor = nargv[optind];
    current_optchar = parse_long_options(nargv, options, long_options, idx, 0);
    option_cursor = EMSG;
    return (current_optchar);
  }
  if (*++option_spec != ':') {
    if (!*option_cursor) ++optind;
  } else {
    optarg = NULL;
    if (*option_cursor)
      optarg = option_cursor;
    else if (option_spec[1] != ':') {
      if (++optind >= nargc) {
        option_cursor = EMSG;
        if (PRINT_ERROR) warnx(recargchar, current_optchar);
        optopt = current_optchar;
        return (BADARG);
      } else
        optarg = nargv[optind];
    }
    option_cursor = EMSG;
    ++optind;
  }

  return (current_optchar);
}

int getopt_long(int nargc, char *const *nargv, const char *options,
                const struct option *long_options, int *idx) {
  return (
      getopt_internal(nargc, nargv, options, long_options, idx, FLAG_PERMUTE));
}

int getopt_long_only(int nargc, char *const *nargv, const char *options,
                     const struct option *long_options, int *idx) {
  return (getopt_internal(nargc, nargv, options, long_options, idx,
                          FLAG_PERMUTE | FLAG_LONGONLY));
}

#ifndef HAVE_DECL_GETOPT
#define HAVE_DECL_GETOPT 1
#endif

#ifdef __cplusplus
}
#endif

#endif