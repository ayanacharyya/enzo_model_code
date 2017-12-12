#!/usr/bin/env python

execfile('filterstars.py')
'''
#print np.min(dd['young_stars', 'particle_mass'])
#print np.max(dd['young_stars', 'particle_mass'])
p = yt.ParticlePlot(ds, ('young_stars', 'particle_position_x'), ('young_stars', 'particle_position_y'), \
('young_stars', 'particle_mass'))
p.set_unit('particle_position_x','kpc')
p.set_unit('particle_position_y','kpc')
#p.set_unit('age','Myr')
#p.set_xlim(650,660)
p.save()
'''
p = yt.ParticlePlot(ds, ('young_stars', 'particle_mass'), ('young_stars', 'age'))
p.set_unit('particle_mass', 'Msun')
p.set_unit('age', 'Myr')
p.save()
'''
plt.scatter(x, y, c=m, s=2, edgecolors='none')
plt.xlabel('Particle position x (kpc)')
plt.ylabel('Particle position y (kpc)')
cb=plt.colorbar()
cb.set_label('Particle mass (Msun)')
#cb.set_label('Age (Myr)')
#plt.xlim(652,657)
#plt.ylim(652,657)
plt.show()
#plt.savefig('DD0550_young_stars_agecolor.png')
'''
