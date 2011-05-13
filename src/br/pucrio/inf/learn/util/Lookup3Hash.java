package br.pucrio.inf.learn.util;

/**
 * A Java implementation of hashword from lookup3.c by Bob Jenkins.
 * 
 * (<a href="http://burtleburtle.net/bob/c/lookup3.c">original source</a>).
 * 
 * @author eraldof
 * 
 */
public class Lookup3Hash {
	public static int hash(int[] key, int seed) {
		int a, b, c;
		int length = key.length;
		a = b = c = 0xdeadbeef + (length << 2) + seed;

		int i = 0;
		while (length > 3) {
			a += key[i];
			b += key[i + 1];
			c += key[i + 2];

			// mix(a,b,c)... Java needs "out" parameters!!!
			// Note: recent JVMs (Sun JDK6) turn pairs of shifts (needed to do a
			// rotate)
			// into real x86 rotate instructions.
			{
				a -= c;
				a ^= (c << 4) | (c >>> -4);
				c += b;
				b -= a;
				b ^= (a << 6) | (a >>> -6);
				a += c;
				c -= b;
				c ^= (b << 8) | (b >>> -8);
				b += a;
				a -= c;
				a ^= (c << 16) | (c >>> -16);
				c += b;
				b -= a;
				b ^= (a << 19) | (a >>> -19);
				a += c;
				c -= b;
				c ^= (b << 4) | (b >>> -4);
				b += a;
			}

			length -= 3;
			i += 3;
		}

		switch (length) {
		case 3:
			c += key[i + 2]; // fall through
		case 2:
			b += key[i + 1]; // fall through
		case 1:
			a += key[i + 0]; // fall through
			// final(a,b,c);
			{
				c ^= b;
				c -= (b << 14) | (b >>> -14);
				a ^= c;
				a -= (c << 11) | (c >>> -11);
				b ^= a;
				b -= (a << 25) | (a >>> -25);
				c ^= b;
				c -= (b << 16) | (b >>> -16);
				a ^= c;
				a -= (c << 4) | (c >>> -4);
				b ^= a;
				b -= (a << 14) | (a >>> -14);
				c ^= b;
				c -= (b << 24) | (b >>> -24);
			}
		case 0:
			break;
		}
		return c;
	}

	public static int hash(byte[] key, int seed) {
		int a, b, c;
		int length = key.length;
		a = b = c = 0xdeadbeef + length + seed;

		int i = 0;
		while (length >= 12) {
			a += ((key[i + 0] & 0xFF) << 24) | ((key[i + 1] & 0xFF) << 16)
					| ((key[i + 2] & 0xFF) << 8) | ((key[i + 3] & 0xFF) << 0);
			b += ((key[i + 4] & 0xFF) << 24) | ((key[i + 5] & 0xFF) << 16)
					| ((key[i + 6] & 0xFF) << 8) | ((key[i + 7] & 0xFF) << 0);
			c += ((key[i + 8] & 0xFF) << 24) | ((key[i + 9] & 0xFF) << 16)
					| ((key[i + 10] & 0xFF) << 8) | ((key[i + 11] & 0xFF) << 0);

			// mix(a,b,c)... Java needs "out" parameters!!!
			// Note: recent JVMs (Sun JDK6) turn pairs of shifts (needed to do a
			// rotate)
			// into real x86 rotate instructions.
			{
				a -= c;
				a ^= (c << 4) | (c >>> -4);
				c += b;
				b -= a;
				b ^= (a << 6) | (a >>> -6);
				a += c;
				c -= b;
				c ^= (b << 8) | (b >>> -8);
				b += a;
				a -= c;
				a ^= (c << 16) | (c >>> -16);
				c += b;
				b -= a;
				b ^= (a << 19) | (a >>> -19);
				a += c;
				c -= b;
				c ^= (b << 4) | (b >>> -4);
				b += a;
			}

			length -= 12;
			i += 12;
		}

		int af = 0;
		int bf = 0;
		int cf = 0;
		switch (length) {
		case 11:
			af |= ((key[i + 9] & 0xFF) << 8);
		case 10:
			af |= ((key[i + 8] & 0xFF) << 16);
		case 9:
			af |= ((key[i + 7] & 0xFF) << 24);

		case 8:
			bf |= ((key[i + 6] & 0xFF) << 0);
		case 7:
			bf |= ((key[i + 5] & 0xFF) << 8);
		case 6:
			bf |= ((key[i + 4] & 0xFF) << 16);
		case 5:
			bf |= ((key[i + 3] & 0xFF) << 24);

		case 4:
			bf |= ((key[i + 3] & 0xFF) << 0);
		case 3:
			cf |= ((key[i + 2] & 0xFF) << 8);
		case 2:
			cf |= ((key[i + 1] & 0xFF) << 16);
		case 1:
			cf |= ((key[i + 0] & 0xFF) << 24);

			a += af;
			b += bf;
			c += cf;
			// final(a,b,c);
			{
				c ^= b;
				c -= (b << 14) | (b >>> -14);
				a ^= c;
				a -= (c << 11) | (c >>> -11);
				b ^= a;
				b -= (a << 25) | (a >>> -25);
				c ^= b;
				c -= (b << 16) | (b >>> -16);
				a ^= c;
				a -= (c << 4) | (c >>> -4);
				b ^= a;
				b -= (a << 14) | (a >>> -14);
				c ^= b;
				c -= (b << 24) | (b >>> -24);
			}
		}
		return c;
	}
}
