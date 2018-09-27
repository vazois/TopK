#ifndef LOCK_H
#define LOCK_H

typedef volatile char lock_t;
inline void __lock(lock_t *_lock)__attribute__((always_inline));
inline void __unlock(lock_t *_lock)__attribute__((always_inline));
inline int __xchg(lock_t *_lock)__attribute__((always_inline));

inline void __lock(lock_t *_lock){
	while(__xchg(_lock)){
		#if defined(__i386__) || defined(__x86_64__)
        __asm__ __volatile__ ("pause\n");
		#endif
	}
}

inline void __unlock(lock_t *_lock){ *_lock = 0; }

inline int __xchg(lock_t *_lock)
{
	register lock_t _r = 1;
	#if defined(__i386__) || defined(__x86_64__)
		__asm__ __volatile__ (
				"lock xchgb %0,%1"
				: "+q"(_r), "+m"(*_lock)
				: /*no input*/
				: "memory" , "cc"
		);
	#else
	#error XCHB not defined for this architecture.
	#endif
	return _r;
}



#endif
